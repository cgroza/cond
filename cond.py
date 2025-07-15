import bed_reader
import zstandard as zstd
import sys
import numpy
import pandas
from sklearn import linear_model
from scipy.stats import linregress
import pdb

bed_path  = sys.argv[1]
qtl_path  = sys.argv[2]
chrom     = sys.argv[3]
cis_chrom = sys.argv[4]

pheno = sys.argv[5]
pheno_path = sys.argv[6]

covar_path = sys.argv[7]

cis_pval = 1.0e-6
trans_pval = 1.0e-9


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

snps.sort(key = lambda x: x["P"])

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

df.dropna(inplace=True)

# remove effect of covariates
covar_reg = linear_model.LinearRegression()
covar_reg.fit(df.iloc[:, 2:22], df['pheno'])

df = df.assign(pheno_corrected = df['pheno'] - covar_reg.predict(df.iloc[:, 2:22]))

pheno_geno_df = pandas.merge(df.loc[:, ["IID", "pheno_corrected"]], bed_data, on='IID', how='inner')


# iterate through snps and condition
snp_order = 0
for snp in snps:
    # skip other chromosomes
    if snp['#CHROM'] != chrom:
        continue
    snp_reg = linregress(pheno_geno_df[snp['ID']], pheno_geno_df['pheno_corrected'], alternative='two-sided')
    snp["p.J"] = snp_reg.pvalue.item()
    snp["BETA.J"] = snp_reg.slope.item()
    snp["SE.J"] = snp_reg.stderr.item()
    snp["order.J"] = snp_order
    snp["sig.J"] = False
    snp_order += 1

    # remove effect of this SNP 
    pheno_geno_df['pheno_corrected'] = pheno_geno_df['pheno_corrected'] - snp_reg.slope * pheno_geno_df[snp['ID']] + snp_reg.intercept

pandas.DataFrame.from_dict(snps).to_csv(sys.stdout, index=False, sep = "\t")
