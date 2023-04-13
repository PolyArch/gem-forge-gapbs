import metis
import sys
import math
import os
import argparse
import random

def scan_vertex(fn):
    min_vertex = -1
    max_vertex = 0
    unique_vertex = set()
    unique_edge = set()
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            unique_vertex.add(src)
            unique_vertex.add(dst)
            unique_edge.add((src, dst))
            if min_vertex == -1:
                min_vertex = min(src, dst)
            else:
                min_vertex = min(min_vertex, src, dst)
            max_vertex = max(max_vertex, src, dst)

    print(f'min = {min_vertex} max = {max_vertex} unique vertex = {len(unique_vertex)} unique edge = {len(unique_edge)}')
    return min_vertex, max_vertex


def read_edge_list_as_adjlist(fn):
    adjlist = list()
    edge_properties = list()
    min_vertex = -1
    max_vertex = 0
    with open(fn) as f:
        for line in f:
            if line.startswith('#') or line.startswith('%'):
                continue
            fields = line.split()
            src = int(fields[0])
            dst = int(fields[1])
            if len(fields) > 2:
                # Some edge properties, usually weight.
                edge_properties.append((src, dst, ' '.join(fields[2:])))
            if min_vertex == -1:
                min_vertex = min(src, dst)
            else:
                min_vertex = min(min_vertex, src, dst)
            max_vertex = max(max_vertex, src, dst)
            while len(adjlist) <= max_vertex:
                adjlist.append(list())
            adjlist[src].append(dst)
    print(f'min = {min_vertex} max = {max_vertex} {len(adjlist)}')
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
    
    # Construct the edge property map and also subtract min_vertex.
    edge_property_map = dict()
    for src, dst, prop in edge_properties:
        u = src - min_vertex if min_vertex > 0 else src
        v = dst - min_vertex if min_vertex > 0 else dst
        if u not in edge_property_map:
            edge_property_map[u] = dict()
        edge_property_map[u][v] = prop

    return (adjlist, min_vertex, edge_property_map)

def get_partition_list(parts):
    part_sets = list()
    for v in range(len(parts)):
        p = parts[v]
        while len(part_sets) <= p:
            part_sets.append(list())
        part_sets[p].append(v)
    return part_sets

def analyze_partition(adjlist, parts, out_fn):
    """
    Print out number of edges across partitions.
    """
    num_edges = sum([len(x) for x in adjlist])
    print(f'TotalDirEdges {num_edges}')
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
    total_edges = 0
    total_internal_edges = 0
    for pid in range(num_parts):
        part = part_sets[pid]
        internal_nodes = 0
        internal_edges = 0
        edges = 0
        for v in part:
            v_pid = parts[v]
            external_edges = 0
            for u in adjlist[v]:
                edges += 1
                if u < 0 or u >= len(parts):
                    print(f'Illegal u {u} Parts {len(parts)}')
                u_pid = parts[u]
                if u_pid != v_pid:
                    external_edges += 1
                else:
                    internal_edges += 1
            if external_edges == 0:
                internal_nodes += 1
        total_internal_nodes += internal_nodes
        total_internal_edges += internal_edges
        total_edges += edges
        print(f'  Part {pid} Nodes {len(part)} InternalNodes {internal_nodes} TotalEdges {edges} InternalEdges {internal_edges}')
    print(f'TotalInterlNodes {total_internal_nodes} Ratio {total_internal_nodes / len(adjlist)}')
    print(f'TotalInterlEdges {total_internal_edges} TotalEdges {total_edges} Ratio {total_internal_edges / total_edges}')

    with open(out_fn, 'w') as f:
        f.write(f'PartSize {num_parts} ')
        for pid in range(num_parts):
            part_size = len(part_sets[pid])
            f.write(f'{part_size} ')
        f.write('\n')
        f.write(f'TotalEdges {total_edges}\n')
        f.write(f'TotalInternalEdges {total_internal_edges}\n')
        f.write(f'InternalEdgesRatio {total_internal_edges / total_edges}\n')

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

def dump_edge_list(fn, edges, edge_properties):
    with open(fn, 'w') as f:
        if edge_properties:
            assert(len(edge_properties) == len(edges))
            sorted_edges = sorted(zip(edges, edge_properties))
            for (u, v), prop in sorted_edges:
                f.write(f'{u} {v} {prop}\n')
        else:
            sorted_edges = sorted(edges)
            for u, v in sorted_edges:
                f.write(f'{u} {v}\n')

def dump_partition(original_fn,
    n_parts,
    adjlist,
    parts,
    is_weight,
    is_symmetry,
    edge_property_map,
    min_vertex,
    part_method,
    ):

    partitions = get_partition_list(parts)
    # Reordered graph nodes starts with 1.
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
    reordered_edge_properties = list()
    for u in range(len(adjlist)):
        reordered_u = reorder_map[u]
        for v in adjlist[u]:
            reordered_v = reorder_map[v]
            reordered_edge_list.append((reordered_u, reordered_v))
            if edge_property_map:
                reordered_edge_properties.append(edge_property_map[u][v])

    prefix = original_fn[:original_fn.rfind('.')]
    fn = f'{prefix}-{part_method}{n_parts}'
    if is_weight:
        el_fn = f'{prefix}-{part_method}{n_parts}.wel'
        el_backup_fn = f'{prefix}-{part_method}{n_parts}.backup.wel'
    else:
        el_fn = f'{prefix}-{part_method}{n_parts}.el'
        el_backup_fn = f'{prefix}-{part_method}{n_parts}.backup.el'

    dump_edge_list(el_fn, reordered_edge_list, reordered_edge_properties)
    dump_edge_list(el_backup_fn, reordered_edge_list, reordered_edge_properties)

    original_src_fn = f'{prefix}.src.txt'
    try:
        f = open(original_src_fn)
    except FileNotFoundError:
        print('No Source File. Pick 0')
        src = 0
    else:
        with f:
            src = int(next(f))

    if min_vertex > 0:
        print(f'Subtract {min_vertex} from Src {src}')
        src = src - min_vertex
    
    reordered_src = reorder_map[src]
    print(f'Source {src} Reordered {reordered_src}')

    # Generate the serialized undirected version, with the same source.
    cmd = f'./converter -f {el_fn} -b {fn} -r {reordered_src}'
    if is_weight:
        cmd += ' -w'
    if is_symmetry:
        cmd += ' -s'
    os.system(cmd)

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

def original_partition(adjlist, nparts):
    num_nodes = len(adjlist)
    parts = [0] * num_nodes
    part_size = (num_nodes + nparts - 1) // nparts
    for i in range(num_nodes):
        parts[i] = i // part_size
        assert(parts[i] < nparts)
    return parts

def random_partition(adjlist, nparts):
    parts = original_partition(adjlist, nparts)
    random.shuffle(parts)
    return parts

def main(argv):

    parser = argparse.ArgumentParser(
        prog = 'Partition Graph')

    parser.add_argument('fn')           # file name for graph.
    parser.add_argument('--nparts',
        type=int, default=64,
        help='Number of partition')     
    parser.add_argument('--symmetric',
        action='store_true', default=False,
        help='Symetric')      
    parser.add_argument('--part',
        choices=['metis', 'orig', 'rnd'], default=metis,
        help='How to partition the graph')      
    parser.add_argument('--out-fn', '-o',
        type=str, default="float-trace.csv",
        help='Output filename')    

    args = parser.parse_args(argv)
    is_weight = args.fn.endswith('.wel')
    adjlist, min_vertex, edge_property_map = read_edge_list_as_adjlist(args.fn)
    # First dump the original version and pick up a source.
    prefix = args.fn[:args.fn.rfind('.')]
    # dump_adjlist(adjlist, prefix, symmetry)

    print(f'Begin partition {args.part}')
    if args.part == 'metis':
        edge_cuts, parts = metis.part_graph(adjlist, args.nparts)
    elif args.part == 'orig':
        parts = original_partition(adjlist, args.nparts)
    elif args.part == 'rnd':
        parts = random_partition(adjlist, args.nparts)

    print('End partition')
    # edge_cuts, parts = bounded_dfs(adjlist, nparts)
    part_fn = f'{prefix}-{args.part}{args.nparts}.part.txt'
    analyze_partition(adjlist, parts, out_fn=part_fn)
    dump_partition(args.fn,
        args.nparts,
        adjlist,
        parts,
        is_weight,
        args.symmetric,
        edge_property_map,
        min_vertex=min_vertex,
        part_method=args.part,
        )


if __name__ == '__main__':
    main(sys.argv[1:])
