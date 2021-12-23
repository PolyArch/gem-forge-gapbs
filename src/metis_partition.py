import metis
import sys
import math
import os

def read_edge_list_as_adjlist(fn):
    adjlist = list()
    min_vertex = -1
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            if min_vertex == -1:
                min_vertex = min(src, dst)
            else:
                min_vertex = min(min_vertex, src, dst)
            while len(adjlist) <= src:
                adjlist.append(list())
            adjlist[src].append(dst)
    if min_vertex > 0:
        print(f'Subtract minimum vertex {min_vertex} from edge list.')
        for i in range(min_vertex):
            if adjlist[i]:
                print(f'Vertex {i} < MinVertex {min_vertex} should have no adjacent nodes.')
                assert(False)
        adjlist = adjlist[min_vertex:]
        for i in range(len(adjlist)):
            for j in range(len(adjlist[i])):
                adjlist[i][j] -= min_vertex
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
    """
    Print out number of edges across partitions.
    """
    num_edges = sum([sum(x) for x in adjlist]) / 2
    print(f'EdgeCut {edge_cuts} TotalEdges {num_edges}')
    """
    Analyze each average parition size and variation.
    """
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
    """
    Analyze each partition.
    """
    total_internal_nodes = 0
    for pid in range(num_parts):
        part = part_sets[pid]
        internal_nodes = 0
        for v in part:
            v_pid = parts[v]
            externl_edges = 0
            for u in adjlist[v]:
                u_pid = parts[u]
                if u_pid != v_pid:
                    externl_edges += 1
            if externl_edges == 0:
                internal_nodes += 1
        total_internal_nodes += internal_nodes
        print(f'  Part {pid} Nodes {len(part)} InternalNodes {internal_nodes}')
    print(f'TotalInterlNodes {total_internal_nodes} Ratio {total_internal_nodes / len(adjlist)}')

def dump_adjlist(adjlist, prefix, symmetry):
    el_fn = f'{prefix}.el'
    with open(el_fn, 'w') as f:
        for u in range(len(adjlist)):
            for v in adjlist[u]:
                f.write(f'{u} {v}\n')
    if symmetry:
        os.system(f'./converter -f {el_fn} -b {prefix} -s')
        os.system(f'./converter -f {el_fn} -b {prefix} -w -s')
    else:
        os.system(f'./converter -f {el_fn} -b {prefix}')
        os.system(f'./converter -f {el_fn} -b {prefix} -w')

def dump_partition(original_fn, n_parts, adjlist, parts):

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

    fn = f'{prefix}-metis{n_parts}'
    el_fn = f'{prefix}-metis{n_parts}.el'
    with open(el_fn, 'w') as f:
        for reordered_u, reordered_v in reordered_edge_list:
            f.write(f'{reordered_u} {reordered_v}\n')

    original_src_fn = f'{prefix}.src.txt'
    try:
        f = open(original_src_fn)
    except FileNotFoundError:
        print('No Source File. Pick 0')
        src = 0
    else:
        with f:
            src = int(next(f))
    
    reordered_src = reorder_map[src]

    # Generate the serialized undirected version, with the same source.
    os.system(f'./converter -f {el_fn} -b {fn} -r {reordered_src}')

def bounded_dfs(adjlist, bounded_depth):
    unclustered_nodes = set()
    clustered_nodes = dict()
    num_nodes = len(adjlist)
    clusters = list()
    cur_cluster_id = -1
    for i in range(num_nodes):
        unclustered_nodes.add(i)
    while len(clustered_nodes) < num_nodes:
        stack = list()
        stack.append((next(iter(unclustered_nodes)), 0))

        while stack:
            v, depth = stack.pop()
            if v in clustered_nodes:
                # Visited.
                continue

            if depth == 0:
                # New clusters.
                clusters.append(list())
                cur_cluster_id += 1
        
            unclustered_nodes.remove(v)
            clustered_nodes[v] = cur_cluster_id
            clusters[cur_cluster_id].append(v)

            for u in adjlist[v]:
                if u not in clustered_nodes:
                    if depth + 1 < bounded_depth:
                        stack.append((u, depth + 1))
                    else:
                        stack.insert(0, (u, 0))
    edge_cuts = 0
    for u in range(num_nodes):
        u_cluster_id = clustered_nodes[u]
        for v in adjlist[u]:
            v_cluster_id = clustered_nodes[v]
            if u_cluster_id != v_cluster_id:
                edge_cuts += 1
    return (edge_cuts, clustered_nodes)



def main(argv):
    fn = argv[1]
    nparts = int(argv[2])
    symmetry = False
    if len(argv) > 3:
        if argv[3] == '-s':
            symmetry = True
    adjlist = read_edge_list_as_adjlist(fn)
    # First dump the original version and pick up a source.
    prefix = fn[:fn.rfind('.')]
    # dump_adjlist(adjlist, prefix, symmetry)

    # edge_cuts, parts = metis.part_graph(adjlist, nparts)
    edge_cuts, parts = bounded_dfs(adjlist, nparts)
    analyze_partition(adjlist, edge_cuts, parts)
    # dump_partition(fn, nparts, adjlist, parts)


if __name__ == '__main__':
    main(sys.argv)
