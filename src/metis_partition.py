import metis
import sys
import math
import os

def read_edge_list_as_adjlist(fn):
    adjlist = list()
    with open(fn) as f:
        for line in f:
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            while len(adjlist) <= src:
                adjlist.append(list())
            adjlist[src].append(dst)
    return adjlist

def get_partition_list(parts):
    part_sets = list()
    for v in range(len(parts)):
        p = parts[v]
        while len(part_sets) <= p:
            part_sets.append(list())
        part_sets[p].append(v)
    return part_sets

def analyze_partition(adjlist, edge_cuts, parts):
    num_edges = sum([sum(x) for x in adjlist]) / 2
    print(f'EdgeCut {edge_cuts} TotalEdges {num_edges}')
    num_parts = max(parts) + 1
    part_sets = get_partition_list(parts)
    avg_part_size = len(adjlist) / num_parts
    std_part_size = 0
    for p in part_sets:
        s = len(p)
        diff = s - avg_part_size
        std_part_size += diff * diff
    std_part_size = math.sqrt(std_part_size)
    print(f'NumParts {num_parts} AvgSize {avg_part_size} Std {std_part_size}')

def dump_to_edge_list(original_fn, n_parts, adjlist, parts):
    partitions = get_partition_list(parts)
    partition_acc = [0] * len(partitions)
    for i in range(1, len(partitions)):
        partition_acc[i] = partition_acc[i - 1] + len(partitions[i - 1])
    reorder_map = dict()
    for v in range(len(parts)):
        p = parts[v]
        reordered_v = partition_acc[p]
        partition_acc[p] += 1
        reorder_map[v] = reordered_v

    reordered_edge_list = list()
    for u in range(len(adjlist)):
        reordered_u = reorder_map[u]
        for v in adjlist[u]:
            reordered_v = reorder_map[v]
            reordered_edge_list.append((reordered_u, reordered_v))

    prefix = original_fn[:original_fn.rfind('.')]
    fn = f'{prefix}-metis{n_parts}'
    el_fn = f'{prefix}-metis{n_parts}.el'
    with open(el_fn, 'w') as f:
        for reordered_u, reordered_v in reordered_edge_list:
            f.write(f'{reordered_u} {reordered_v}\n')

    original_src_fn = f'{prefix}.src.txt'
    with open(original_src_fn) as f:
        src = int(next(f)) 
    
    reordered_src = reorder_map[src]

    # Generate the serialized undirected version, with the same source.
    os.system(f'./converter -s -f {el_fn} -b {fn} -r {reordered_src}')


def main(argv):
    fn = argv[1]
    nparts = int(argv[2])
    adjlist = read_edge_list_as_adjlist(fn)
    edge_cuts, parts = metis.part_graph(adjlist, nparts)
    analyze_partition(adjlist, edge_cuts, parts)
    dump_to_edge_list(fn, nparts, adjlist, parts)


if __name__ == '__main__':
    main(sys.argv)