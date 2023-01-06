import os

n_start = 10
n_end = 21

for n in range(21, 22):
    for k in [8]:
        os.system(f'./converter -u {n} -k {k} -b benchmark/graphs/uni{n}-k{k}')
        os.system(f'./converter -g {n} -k {k} -b benchmark/graphs/krn{n}-k{k}')
        # for nparts in [16, 32, 64, 128, 256, 512, 1024]:
        #     os.system(f'python src/metis_partition.py benchmark/graphs/uni{n}-k{k}.el {nparts}')
        #     os.system(f'python src/metis_partition.py benchmark/graphs/krn{n}-k{k}.el {nparts}')
