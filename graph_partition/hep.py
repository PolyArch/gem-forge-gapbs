import graph_tool.all as gt
import math
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, List, Dict

'''Graph Terminology
Order = number of nodes
Size = number of edges
'''


class PartitionGraph:
    @staticmethod
    def from_edge_partition(g: gt.Graph, edge_membership: gt.EdgePropertyMap,
                            num_partitions: int)\
            -> Tuple[gt.Graph, gt.Graph, gt.VertexPropertyMap, List]:
        '''From an edge membership, return partitioned graph with duplicate
        nodes.

        '''

        node_membership = np.zeros(
            (g.num_vertices(), num_partitions), dtype=bool)

        partition_sizes = [0] * num_partitions

        for (src_id, dst_id) in g.iter_edges():
            partition_id = edge_membership[g.edge(src_id, dst_id)]
            node_membership[src_id][partition_id] = True
            node_membership[dst_id][partition_id] = True
            partition_sizes[partition_id] += 1

        # New graph with duplicate nodes
        new_graph = g.copy()
        # Duplicate nodes point to original node
        partition_graph = gt.Graph(directed=True)

        multi_membership_mask = np.sum(node_membership, axis=1) > 1
        multi_membership_ids = np.where(multi_membership_mask)[0]

        # Generate duplicate nodes.
        NodeId = int
        PartitionId = int
        multi_membership_nodes: Dict[NodeId, Dict[PartitionId, gt.Vertex]] =\
            defaultdict(dict)

        for node_id in multi_membership_ids:
            new_graph.clear_vertex(node_id)  # Remove edges

            # Make duplicates.
            membership = node_membership[node_id].flatten().nonzero()[0]
            duplicate_count = len(membership) - 1
            duplicate_nodes = new_graph.add_vertex(duplicate_count)
            nodes = [new_graph.vertex(node_id)]
            if duplicate_count == 1:
                nodes.append(duplicate_nodes)
            else:
                nodes += list(duplicate_nodes)

            # Update partition map.
            for partition_id, node in zip(membership, nodes):
                multi_membership_nodes[node_id][partition_id] = node

            # Update partition graph.
            original_node = partition_graph.vertex(node_id, add_missing=True)
            for duplicate in nodes[1:]:
                duplicate_id = int(duplicate)
                duplicate_node = partition_graph.vertex(
                    duplicate_id, add_missing=True)
                partition_graph.add_edge(duplicate_node, original_node)

        # Remap nodes & edges appropriatedly.
        # We keep the original node and add n-1 duplicates
        for node_id in multi_membership_ids:
            # Add new edges corresponding to each partition.
            neighbor_edges = g.get_all_edges(node_id, eprops=[edge_membership])
            for partition_id, duplicate_node in \
                    multi_membership_nodes[node_id].items():
                edge_mask = neighbor_edges[:, 2] == partition_id
                for id1, id2, _ in neighbor_edges[edge_mask]:
                    other_id = id1 if id1 != node_id else id2
                    if other_id in multi_membership_nodes:
                        other_node = \
                            multi_membership_nodes[other_id][partition_id]
                    else:
                        other_node = other_id
                    new_graph.edge(duplicate_node, other_node,
                                   add_missing=True)

        # Relabel nodes based on partition
        new_node_membership = new_graph.new_vertex_property('int')
        for node_id in g.iter_vertices():
            membership = node_membership[node_id].flatten().nonzero()[0]
            assert len(membership) != 0
            if len(membership) == 1:  # Is only in one partition
                partition_id = membership[0]
                new_node_membership[node_id] = partition_id
            else:
                for partition_id in membership:
                    dup_node = multi_membership_nodes[node_id][partition_id]
                    new_node_membership[dup_node] = partition_id

        new_node_remap = new_graph.new_vertex_property('int')
        partition_order_prefix = 0  # Order = number of nodes in a graph
        partition_orders: List[int] = [0] * num_partitions
        for partition_id in range(num_partitions):
            mask = new_node_membership.get_array() == partition_id
            order = np.count_nonzero(mask)
            new_ids = np.arange(order) + partition_order_prefix
            new_node_remap.get_array()[mask] = new_ids

            partition_orders[partition_id] = order
            partition_order_prefix += order

        partition_node_remap = partition_graph.new_vertex_property('int')
        for node_id in partition_graph.iter_vertices():
            partition_node_remap[node_id] = new_node_remap[node_id]

        relabeled_new_graph = gt.Graph(new_graph, vorder=new_node_remap)
        relabeled_partition_graph = gt.Graph(
            partition_graph, vorder=partition_node_remap)

        return relabeled_new_graph, relabeled_partition_graph, partition_orders, partition_sizes


class HEP:
    '''Hybrid Edge Partitioner
    https://dl.acm.org/doi/10.1145/3448016.3457300
    '''

    def __print_node(self, stage: str, node_id: int):
        print("[{stage}] Node {node_id} Degree {node_degree}".format(
            stage=stage,
            node_id=node_id,
            node_degree=g.get_out_degrees([node_id])[0]
        ))

    def __init__(self, seed=None):
        self._generator = np.random.default_rng(seed=seed)

    def partition(self, g: gt.Graph, num_partitions: int)\
            -> gt.EdgePropertyMap:
        '''Returns a map of each edge's membership.'''

        self._removed_edges = g.new_edge_property('bool')
        self._removed_edges.get_array()[:] = False

        self._g = gt.GraphView(g)
        self._partition_size = math.floor(g.num_edges() / num_partitions)

        # Partition id is 1-indexed (0 = not assigned)
        self._edge_partition = self._g.new_edge_property('int')
        self._num_partitions = num_partitions
        self._core = np.zeros(self._g.num_vertices(), dtype=bool)

        self._secondary = np.zeros(self._g.num_vertices(), dtype=bool)
        self._ext_degree = np.zeros(self._g.num_vertices(), dtype=int)

        self._spilled_nodes: set(int) = set()
        self._spilled_edges: List[Tuple[int, int]] = list()

        self._partition_graph()

        # Set partition ids to zero-indexed.
        edge_membership = self._edge_partition.copy()
        edge_membership.get_array()[:] -= 1
        return edge_membership

    def _cur_partition_size(self, partition_id: int) -> bool:
        edge_partition_asnp = self._edge_partition.get_array()
        edge_count = np.count_nonzero(edge_partition_asnp == partition_id)
        return edge_count

    def _is_partition_full(self, partition_id: int) -> bool:
        return self._cur_partition_size(partition_id) >= self._partition_size

    def _initialize(self, partition_id: int):
        node_id = self._generator.choice(self._g.get_vertices()[~self._core])
        self._move_to_core(node_id, partition_id)

    def _min_ext_degree_id(self) -> int:
        max_value = self._ext_degree.max() + 1
        self._ext_degree[~self._secondary] = max_value
        min_node_id = self._ext_degree.argmin()
        self._ext_degree[~self._secondary] = 0
        return min_node_id

    def _has_spillover(self) -> bool:
        assert (len(self._spilled_nodes) > 0) == (len(self._spilled_edges) > 0)
        return len(self._spilled_nodes) > 0

    def _handle_spillover(self, partition_id: int):
        self._secondary[:] = False
        self._ext_degree[:] = 0

        # Set new nodes & ext degree.
        spilled_node_ids = np.array(list(self._spilled_nodes))
        self._secondary[spilled_node_ids] = True
        self._ext_degree[spilled_node_ids] = g.get_total_degrees(
            spilled_node_ids)

        # Add edges and recompute
        assert not self._is_partition_full(partition_id)
        for commit_idx, (src_id, dst_id) in enumerate(self._spilled_edges):
            if not self._is_partition_full(partition_id):
                assert self._ext_degree[src_id] >= 0
                assert self._ext_degree[dst_id] >= 0
                self._ext_degree[src_id] -= 1
                self._ext_degree[dst_id] -= 1
                edge = self._g.edge(src_id, dst_id, add_missing=False)
                assert self._edge_partition[edge] == 0
                self._edge_partition[edge] = partition_id
                self._remove_edge(edge)
            else:
                commit_idx -= 1
                break

        if commit_idx != len(self._spilled_edges) - 1:
            # If spilled again (i.e., spilled edges > partition size)
            # Save remaining edges and recompute spilled nodes
            # commit_idx = commit up till this index (inclusive)
            remaining_edges = self._spilled_edges[commit_idx + 1:]
            self._spilled_nodes.clear()
            for src_id, dst_id in remaining_edges:
                self._spilled_nodes.add(src_id)
                self._spilled_nodes.add(dst_id)
            self._spilled_edges = remaining_edges
        else:
            self._spilled_nodes.clear()
            self._spilled_edges.clear()

    def _partition_graph(self):
        # Partition according to buckets.
        for partition_id in (np.arange(self._num_partitions) + 1):
            progress_bar = tqdm(total=self._partition_size,
                                desc=f'Partition {partition_id}')
            prev_partition_size = 0
            if self._has_spillover():
                self._handle_spillover(partition_id)

            cur_partition_size = self._cur_partition_size(partition_id)
            progress_bar.update(cur_partition_size - prev_partition_size)
            prev_partition_size = cur_partition_size

            iter = 0
            while not self._is_partition_full(partition_id):
                excl_in_secondary = self._secondary & ~self._core
                if np.count_nonzero(excl_in_secondary) != 0:
                    self._move_to_core(self._min_ext_degree_id(), partition_id)
                else:
                    self._initialize(partition_id)

                iter += 1
                if iter % 10 == 0:
                    cur_partition_size = self._cur_partition_size(partition_id)
                    progress_bar.update(
                        cur_partition_size - prev_partition_size)
                    prev_partition_size = cur_partition_size

            cur_partition_size = self._cur_partition_size(partition_id)
            progress_bar.update(cur_partition_size - prev_partition_size)
            prev_partition_size = cur_partition_size
            progress_bar.close()

        # Assign remaining edges to final edge partition
        print("Unassigned count", self._cur_partition_size(0))
        edge_partition = self._edge_partition.get_array()
        unassigned_edges = edge_partition == 0
        edge_partition[unassigned_edges] = self._num_partitions

    def _move_to_core(self, node_id: int, partition_id: int):
        self._core[node_id] = True
        self._secondary[node_id] = False  # Remove from secondary
        neighbors = self._as_node_mask(self._g.get_out_neighbors(node_id))
        exclude = self._core | self._secondary
        secondary_neighbors = neighbors & ~exclude

        for neighbor_id in g.get_vertices()[secondary_neighbors]:
            self._move_to_secondary(neighbor_id, partition_id)

    def _move_to_secondary(self, node_id: int, partition_id: int):
        self._secondary[node_id] = True
        self._ext_degree[node_id] = self._g.get_total_degrees([node_id])

        neighbors = self._as_node_mask(self._g.get_out_neighbors(node_id))
        include = self._core | self._secondary
        partition_neighbors = neighbors & include

        for neighbor_id in g.get_vertices()[partition_neighbors]:
            edge = self._g.edge(node_id, neighbor_id, add_missing=False)

            if not self._is_partition_full(partition_id):
                self._ext_degree[node_id] -= 1
                self._ext_degree[neighbor_id] -= 1
                assert self._edge_partition[edge] == 0
                self._edge_partition[edge] = partition_id
                self._remove_edge(edge)
            else:
                self._spilled_nodes.add(node_id)
                self._spilled_nodes.add(neighbor_id)
                self._spilled_edges.append((node_id, neighbor_id))

    def _as_node_mask(self, indices: np.array) -> np.array:
        mask = np.zeros(self._g.num_vertices(), dtype=bool)
        mask[indices] = True
        return mask

    def _remove_edge(self, edge: gt.Edge):
        self._removed_edges[edge] = True
        self._g.set_edge_filter(self._removed_edges, inverted=True)


if __name__ == '__main__':
    import sys
    import pathlib
    # fname_edge_list = 'krn17-k16.el'
    # fname_edge_list = 'kron-14-16.el'
    fname_edge_list = sys.argv[1]
    fname_graph = str(pathlib.Path(fname_edge_list).with_suffix('.gt'))

    # Construct graph from edge list
    edge_list = np.loadtxt(fname_edge_list, dtype='int')
    g = gt.Graph(directed=False)
    g.add_edge_list(edge_list)
    gt.remove_parallel_edges(g)

    # Prune isolated nodes
    degs = g.get_total_degrees(g.get_vertices())
    isolated_nodes = degs == 0
    g = gt.GraphView(g, vfilt=~isolated_nodes)
    g = gt.Graph(g, prune=True)

    # Save graph to file
    g.save(fname_graph)

    # g = gt.load_graph(fname_graph)
    # g.save(fname_graph)
    # print(g)

    # if False:
    #     g = gt.Graph(directed=False)
    #     g.add_edge_list([
    #         [0, 3], [1, 3], [2, 3], [0, 1], [1, 2], [0, 2],
    #         [4, 3], [5, 3], [6, 3], [4, 5], [5, 6], [4, 6]])
    #     # g = gt.complete_graph(10)

    partitions = int(sys.argv[2])

    hep = HEP()
    edge_membership = hep.partition(g, partitions)
    new_g, partition_g, partition_orders, partition_sizes = \
        PartitionGraph.from_edge_partition(g, edge_membership, partitions)

    # Emit edge lists
    fname_base = str(pathlib.Path(fname_edge_list).with_suffix(''))
    fname_el = f'{fname_base}-ne{partitions}.el'
    fname_partition_el = f'{fname_base}-ne{partitions}.part.el'
    fname_order = f'{fname_base}-ne{partitions}.part.txt'

    with open(fname_el, 'w') as ostream:
        for src, dst in new_g.iter_edges():
            ostream.write(f'{src} {dst}\n')

    with open(fname_partition_el, 'w') as ostream:
        for src, dst in partition_g.iter_edges():
            ostream.write(f'{src} {dst}\n')

    with open(fname_order, 'w') as ostream:
        ostream.write(f'PartSize {partitions}')
        for order in partition_orders:
            ostream.write(f' {order}')
        ostream.write('\n')
        total_order = sum(partition_orders)
        ostream.write(f'Total Nodes {total_order}\n')
        ostream.write(f'PartEdgeSize {partitions}')
        for size in partition_sizes:
            ostream.write(f' {size}')
        ostream.write('\n')
        ostream.write(f'PartEdgePerNode {partitions}')
        for size, order in zip(partition_sizes, partition_orders):
            ostream.write(f' {size/order:.2f}')
        ostream.write('\n')

    # Also dump the per-partition graph.
    all_edges = list(new_g.iter_edges())
    acc_size = 0
    for part in range(len(partition_sizes)):
        size = partition_sizes[part]
        fname_el = f'{fname_base}-ne{partitions}-sub{part}.el'
        with open(fname_el, 'w') as ostream:
            for src, dst, in all_edges[acc_size:acc_size + size]:
                ostream.write(f'{src} {dst}\n')
        acc_size += size
        

