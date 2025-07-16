import bed_reader
import zstandard as zstd
import sys
import numpy
import pandas
from sklearn import linear_model
from scipy.stats import linregress
import pdb
import statsmodels.api as sm


bed_path  = sys.argv[1]
qtl_path  = sys.argv[2]
chrom     = sys.argv[3]
cis_chrom = sys.argv[4]

pheno = sys.argv[5]
pheno_path = sys.argv[6]

covar_path = sys.argv[7]

cis_pval = 1.0e-6
trans_pval = 1.0e-9

pval_threshold = 7.7e-11

if cis_chrom == chrom:
    pval_threshold = 5.e-8

bed = bed_reader.open_bed(bed_path)

snps = []

with zstd.open(qtl_path, 'r') as qtl_file:
    fields = qtl_file.readline().rstrip().split()

    for line in qtl_file:
        snp = dict(zip(fields, line.split()))

        if snp["#CHROM"] != chrom:
            continue

        if snp["P"] == "NA":
            continue

        if snp["#CHROM"] == cis_chrom and float(snp["P"]) < cis_pval:
            snps.append(snp)

        elif snp["#CHROM"] != cis_chrom and float(snp["P"]) < trans_pval:
            snps.append(snp)

snps.sort(key = lambda x: float(x["P"]))

# load genotypes
snp_indices = [numpy.nonzero(bed.sid == snp["ID"])[0][0] for snp in snps]
sample_indices = [numpy.nonzero(bed.iid == sample)[0][0].item() for sample in bed.iid]
bed_data = pandas.DataFrame(bed.read((sample_indices, snp_indices)))
bed_data = bed_data.set_axis([snp["ID"] for snp in snps], axis=1)
bed_data = bed_data.assign(IID = bed.iid)

pheno_values = None

# load pheno measurements
with open(pheno_path, 'r') as pheno_file:
    pheno_dict = {}
    fields = pheno_file.readline().rstrip().split()
    for line in pheno_file:
        pheno_record = dict(zip(fields, line.split()))
        pheno_dict[pheno_record["IID"]] = pheno_record[pheno]
    pheno_values = numpy.array([pheno_dict[sample] for sample in bed.iid])

# load covariates
df = pandas.read_csv(covar_path, sep=',', header=0)
df['IID'] = pandas.Categorical(df['IID'], categories=bed.iid, ordered=True)
df.sort_values(by='IID', inplace=True)

df = df.assign(pheno=pheno_values)
df['pheno'] = pandas.to_numeric(df['pheno'], errors='coerce')


pheno_geno_df = pandas.merge(df, bed_data, on='IID', how='inner')

sig_snps = snps.copy()

covariates = ["Sex.1.Male.2.Female","Age","PC1","PC2","PC3","PC4",
              "PC5","PC6","PC7","PC8","PC9","PC10","PC11","PC12",
              "PC13","PC14","PC15","PC16","PC17","PC18","PC19","PC20"]

output_snps = []

while sig_snps:
    top_snp = sig_snps[0]
    if 'COND.J' not in top_snp:
        top_snp['SE_J']  = top_snp["SE"]
        top_snp['BETA_J']  = top_snp["BETA"]
        top_snp['P_J']  = top_snp["P"]
        top_snp['COND_J']  = "."

    output_snps.append(top_snp)

    new_sig_snps = []

    for snp in sig_snps[1:]:
        # don't condition on SNPs that are more than 10 MBp away
        if abs(int(snp['POS']) - int(top_snp['POS'])) > 1e7:
            new_sig_snps.append(snp)
            continue

    # regress conditioning on the top SNP
        X = pheno_geno_df.loc[:, ['pheno', top_snp['ID'], snp['ID']] + covariates]
        X.dropna(inplace=True)

        model = sm.OLS(X['pheno'], X.loc[:, [top_snp['ID'], snp['ID']] + covariates]).fit()

        snp["SE_J"] = model.bse[snp['ID']]
        snp["BETA_J"] = model.params[snp['ID']]
        snp["P_J"] = model.pvalues[snp['ID']]
        snp["COND_J"] = top_snp['ID']


        if model.pvalues[snp['ID']] < pval_threshold:
            new_sig_snps.append(snp)
        else:
            output_snps.append(snp)

    sig_snps = new_sig_snps
    # resort by adjusted P-value
    sig_snps.sort(key = lambda x: float(x["P_J"]))

pandas.DataFrame.from_dict(output_snps).to_csv(sys.stdout, index=False, sep = "\t")
