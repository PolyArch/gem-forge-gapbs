import os

n_start = 10
n_end = 21

os.system('mkdir -p benchmark/graphs')

# n is 2^n nodes.
for n in range(10, 20):
    # k is average edges/nodes.
    for k in [2, 4, 8, 16]:
        # Generate unweighted uniform graph.
        os.system(f'./converter -u {n} -k {k} -b benchmark/graphs/uni{n}-k{k}')
        # Add weights to the uniform graph.
        os.system(f'./converter -u {n} -k {k} -w -f benchmark/graphs/uni{n}-k{k}.el -b benchmark/graphs/uni{n}-k{k}')
        # Generate unweighted power-law graph.
        os.system(f'./converter -g {n} -k {k} -b benchmark/graphs/krn{n}-k{k}')
        # Add weights to the power-law graph.
        os.system(f'./converter -g {n} -k {k} -w -f benchmark/graphs/krn{n}-k{k}.el -b benchmark/graphs/krn{n}-k{k}')
