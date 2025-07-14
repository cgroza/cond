import bed_reader
import zstandard as zstd
import sys
import numpy

from scipy.stats import linregress

bed_path = sys.argv[1]
qtl_path = sys.argv[2]
cis_chrom = sys.argv[3]

cis_pval = 1.0e-6
trans_pval = 1.0e-9


bed = bed_reader.open_bed(bed_path)

snps = []

with zstd.open(qtl_path, 'r') as qtl_file:
    fields = qtl_file.readline().rstrip().split()

    for line in qtl_file:
        snp = dict(zip(fields, line.split()))

        if snp["P"] == "NA":
            continue

        if snp["#CHROM"] == cis_chrom and float(snp["P"]) < cis_pval:
            snps.append(snp)

        elif snp["#CHROM"] != cis_chrom and float(snp["P"]) < trans_pval:
            snps.append(snp)

snps.sort(key = lambda x: x["P"])

print("Selected", len(snps), "SNPs")
print(bed.sid.index)
snps_indices = [numpy.nonzero(bed.sid == snp["ID"])[0][0] for snp in snps]

bed_data = bed.read_sparse(snps_indices)
print(bed_data)
