import os

n_start = 10
n_end = 21

# for n in range(17, 22):
#     for k in [1, 2]:
        # os.system(f'./converter -u {n} -k {k} -b benchmark/graphs/uni{n}-k{k}')
        # os.system(f'./converter -g {n} -k {k} -b benchmark/graphs/krn{n}-k{k}')
        # os.system(f'./converter -u {n} -k {k} -w -b benchmark/graphs/uni{n}-k{k}')
        # os.system(f'./converter -g {n} -k {k} -w -b benchmark/graphs/krn{n}-k{k}')

def dedup_edge_list(fn, out_fn):
    max_vertex = 0
    unique_edge = set()
    vertex_map = dict()
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            if src not in vertex_map:
                vertex_map[src] = max_vertex
                max_vertex += 1
            if dst not in vertex_map:
                vertex_map[dst] = max_vertex
                max_vertex += 1

            new_src = vertex_map[src]
            new_dst = vertex_map[dst]

            unique_edge.add((new_src, new_dst))

    with open(out_fn, 'w') as f:
        for src, dst in unique_edge:
            f.write(f'{src} {dst}\n')

# dedup_edge_list('benchmark/graphs/ego-fb.txt', 'benchmark/graphs/ego-fb.el')
# dedup_edge_list('benchmark/graphs/ego-twitter.txt', 'benchmark/graphs/ego-twitter.el')
# dedup_edge_list('benchmark/graphs/ego-gplus.txt', 'benchmark/graphs/ego-gplus.el')


for nparts in [64]:
    for method in [
        # 'metis',
        # 'orig',
        'rnd',
        ]:
        # for n in [17, 18, 19, 20, 21]:
        for n in [19]:
            for k in [4]:
                # Do this also for weighted and unweighted graph.
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/krn{n}-k{k}.el --part={method} --nparts={nparts}')
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/krn{n}-k{k}.wel --part={method} --nparts={nparts}')

                # # Also try uniform graph?
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.el --part={method} --nparts={nparts}')
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.wel --part={method} --nparts={nparts}')
                pass

        # Undirected graph.
        # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/twitch-gamers.el --part={method} --nparts={nparts}')
        # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/roadNet-PA.el --part={method} --nparts={nparts}')
        # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/roadNet-TX.el --part={method} --nparts={nparts}')

        # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/ego-fb.el --part={method} --nparts={nparts}')

        # This is directed graph.
        os.system(f'python src/metis_partition.py benchmark/graphs/ego-twitter.el --part={method} --nparts={nparts}')
        # os.system(f'python src/metis_partition.py benchmark/graphs/ego-gplus.el --part={method} --nparts={nparts}')
        # os.system(f'python src/metis_partition.py benchmark/graphs/web-BerkStan.el --part={method} --nparts={nparts}')