import os

n_start = 10
n_end = 21

# for n in range(10, 17):
#     for k in [2]:
#         os.system(f'./converter -u {n} -k {k} -b benchmark/graphs/uni{n}-k{k}')
#         os.system(f'./converter -u {n} -k {k} -w -b benchmark/graphs/uni{n}-k{k}')
        # os.system(f'./converter -g {n} -k {k} -b benchmark/graphs/krn{n}-k{k}')
        # os.system(f'./converter -g {n} -k {k} -w -f benchmark/graphs/krn{n}-k{k}.el -b benchmark/graphs/krn{n}-k{k}')

def dedup_edge_list(fn, out_fn):
    # First pass to find all src vetex.
    src_vertexes = set()
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            src_vertexes.add(src)
    # Second pass to assign vertex.
    max_vertex = 0
    vertex_map = dict()
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            # It's possible that dst is not in the map.
            if src not in vertex_map:
                vertex_map[src] = max_vertex
                max_vertex += 1
            if dst not in vertex_map and dst not in src_vertexes:
                vertex_map[dst] = max_vertex
                max_vertex += 1
    # Third pass to translate edges.
    unique_edge = set()
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            assert(src) in vertex_map
            assert(dst) in vertex_map

            new_src = vertex_map[src]
            new_dst = vertex_map[dst]

            unique_edge.add((new_src, new_dst))

    with open(out_fn, 'w') as f:
        for src, dst in unique_edge:
            f.write(f'{src} {dst}\n')

    print(f'Dedup {fn} -> {out_fn}')

def get_src(fn):
    original_src_fn = f'{fn}.src.txt'
    try:
        f = open(original_src_fn)
    except FileNotFoundError:
        print('No Source File. Pick 0')
        src = 0
    else:
        with f:
            src = int(next(f))
    return src

def part(prefix, method, parts, symmetric):
    if method == 'ne':
        os.system(f'python graph_partition/hep2.py {prefix}.el {nparts}')
        # Generate the serialized graph.
        cmd = f'./converter -f {prefix}-ne{nparts}.el -b {prefix}-ne{nparts}'
        if symmetric:
            cmd += ' -s'
        os.system(cmd)
        # Generate the serialized graph for each sub-graph.
        for i in range(nparts):
            cmd = f'./converter -f {prefix}-ne{nparts}-sub{i}.el -b {prefix}-ne{nparts}-sub{i}'
            if symmetric:
                cmd += ' -s'
            os.system(cmd)
    else:
        if symmetric:
            os.system(f'python src/metis_partition.py --symmetric {prefix}.el --part={method} --nparts={nparts}')
            os.system(f'python src/metis_partition.py --symmetric {prefix}.wel --part={method} --nparts={nparts}')
        else:
            os.system(f'python src/metis_partition.py {prefix}.el --part={method} --nparts={nparts}')
            os.system(f'python src/metis_partition.py {prefix}.wel --part={method} --nparts={nparts}')

folder = 'benchmark/graphs'

# Some edge list need deduplication.
# dedup_edge_list(f'{folder}/ego-fb.txt',      f'{folder}/ego-fb.el')
# dedup_edge_list(f'{folder}/ego-twitter.txt', f'{folder}/ego-twitter.el')
# dedup_edge_list(f'{folder}/ego-gplus.txt',   f'{folder}/ego-gplus.el')
# dedup_edge_list(f'{folder}/twitch-gamers.txt',   f'{folder}/twitch-gamers.el')
# dedup_edge_list(f'{folder}/soc-LiveJournal1.txt',   f'{folder}/soc-LiveJournal1.el')

# Generate weight from unweighted graph.
# os.system(f'./converter -f {folder}/ego-fb.el -w -e {folder}/ego-fb')
# os.system(f'./converter -f {folder}/ego-twitter.el -w -e {folder}/ego-twitter')
# os.system(f'./converter -f {folder}/ego-gplus.el -w -e {folder}/ego-gplus')
# os.system(f'./converter -f {folder}/twitch-gamers.el -w -e {folder}/twitch-gamers')
# os.system(f'./converter -f {folder}/web-BerkStan.el -w -e {folder}/web-BerkStan')
# os.system(f'./converter -f {folder}/soc-LiveJournal1.el -w -e {folder}/soc-LiveJournal1')
# os.system(f'./converter -f {folder}/krn20-k2.el -w -s -e {folder}/krn20-k2')

for nparts in [64]:
    for method in [
        # 'metis',
        # 'rnd',
        'ne',
        # 'orig',
        ]:
        # for n in [17, 18, 19, 20, 21]:
        # for n in [19]:
        #     for k in [4]:
        for x in [1]:
            for n, k in [(15, 64), (16, 32), (18, 8), (19, 4)]:
                # Do this also for weighted and unweighted graph.
                # part(f'{folder}/krn{n}-k{k}', method, nparts)

                # # Also try uniform graph?
                # part(f'{folder}/uni{n}-k{k}', method, nparts)
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.el --part={method} --nparts={nparts}')
                # os.system(f'python src/metis_partition.py --symmetric benchmark/graphs/uni{n}-k{k}.wel --part={method} --nparts={nparts}')
                pass

        # Undirected graph.
        symmetric = True

        # part(f'{folder}/krn20-k2', method, nparts, symmetric)
        # part(f'{folder}/krn19-k4', method, nparts, symmetric)
        # part(f'{folder}/krn18-k8', method, nparts, symmetric)
        # part(f'{folder}/krn17-k16', method, nparts, symmetric)
        # part(f'{folder}/krn16-k32', method, nparts, symmetric)
        # part(f'{folder}/krn15-k64', method, nparts, symmetric)

        part(f'{folder}/twitch-gamers', method, nparts, symmetric)

        # This is directed graph.
        symmetric = False
        # os.system(f'python src/metis_partition.py {folder}/web-BerkStan.el --part={method} --nparts={nparts}')

        # os.system(f'python src/metis_partition.py {folder}/ego-twitter.el --part={method} --nparts={nparts}')

        part(f'{folder}/ego-gplus', method, nparts, symmetric)
        # part(f'{folder}/soc-LiveJournal1', method, nparts, symmetric)
