import os

n_start = 10
n_end = 21

# for n in range(10, 22):
#     for k in [4, 8, 16, 32]:
#         os.system(f'./converter -u {n} -k {k} -w -b benchmark/graphs/uni{n}-k{k}')
#         os.system(f'./converter -g {n} -k {k} -w -b benchmark/graphs/krn{n}-k{k}')
        # for nparts in [16, 32, 64, 128, 256, 512, 1024]:
        #     os.system(f'python src/metis_partition.py benchmark/graphs/uni{n}-k{k}.el {nparts}')
        #     os.system(f'python src/metis_partition.py benchmark/graphs/krn{n}-k{k}.el {nparts}')

for nparts in [64]:
    for n in [14]:
        for k in [16]:
            # Do this also for weighted and unweighted graph.
            os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/krn{n}-k{k}.wel --nparts={nparts}')
            os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/krn{n}-k{k}.el --nparts={nparts}')

            # Also try uniform graph?
            os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.wel --nparts={nparts}')
            os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.el --nparts={nparts}')
            pass

nparts = 64
# os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/roadNet-PA.el --nparts={nparts}')
# os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/roadNet-TX.el --nparts={nparts}')

# This is directed graph.
# os.system(f'python src/metis_partition.py benchmark/graphs/web-BerkStan.el --nparts={nparts}')