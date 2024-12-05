import matplotlib.pyplot as plt

import networkx as nx
import heapq
from itertools import count
import numpy as np
#import numba

# https://gist.github.com/popcornell/bc29d1b7ba37d824335ab7b6280f7fec
#@numba.jit(nopython=True, parallel=True)  # parallel speeds up computation only over very large matrices
# M is a mxn matrix binary matrix
# all elements in M should be uint8
def gf2elim(M):
    m, n = M.shape
    i = 0
    j = 0
    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) + i
        # swap rows
        if M[k, j] == 0:  # Skip if no pivot found
            j += 1
            continue
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp
        # Eliminate below
        for l in range(i + 1, m):
            if M[l, j] == 1:
                M[l, j:] ^= M[i, j:]
        i += 1
        j += 1
    return M

total_number_of_nodes = 0


def print_graph(pg, name, att='capacity'):

    plt.figure()
    pos = nx.spring_layout(pg)  # Position nodes in a visually appealing way
    nx.draw(pg, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_color='black')
    edge_labels = nx.get_edge_attributes(pg, att)
    nx.draw_networkx_edge_labels(pg, pos, edge_labels=edge_labels)
    plt.savefig(name)
    plt.close()


class MinCutEnumerator():
    def run(self, graph, show_all):
        """
        Main method for min cut enumeration

        Parameters:
            - graph -- the graph to process
            - show_all -- defines whether to show all cuts or just cuts which split the graph into 2 parts

        Returns:
            - cut enumeration
        """
        global total_number_of_nodes
        # Load the graph from the specified file
        G = nx.read_graphml(graph)

        # Update graph with integer labels
        label_to_int = {label: i for i, label in enumerate(G.nodes())}
        # Change node labels to integers
        nx.relabel_nodes(G, label_to_int, copy=False)

        for i, (u,v) in enumerate(G.edges(data=False)):    
            G.edges[u,v]['index'] = i

        # Initialize directed graph
        directed_graph = self.convert_to_directed(G)
        total_number_of_nodes = G.number_of_nodes()

        # Basic Partition
        return self.enumerate_min_cuts(G, directed_graph, show_all)

    def convert_to_directed(self, graph):
        """
        Converts a undirected graph to a directed graph

        Parameters:
            - graph -- the graph to transform

        Returns:
            - directed_graph -- the directed graph
        """

        #print_graph(graph,  "Base")

        # Initialize a directed graph
        directed_graph = nx.DiGraph()
        # Add nodes
        directed_graph.add_nodes_from(graph.nodes(data=True))
        # Add directed edges and their reverse, copying the 'order' attribute
        for u, v, data in graph.edges(data=True):
            directed_graph.add_edge(u, v, **data)
            directed_graph.add_edge(v, u, **data)
        return directed_graph

    def hao_orlin(self, graph, source, sink, reset):
        """
        Main function of the Hao-Orlin min cut enumeration algorithm

        Parameters:
            - graph -- the graph to perform the algorithm on
            - source -- the source node
            - sink -- the sink node (if not defined, sink will be determined by the algorithm)
            - reset -- defines whether to perform a reset on the graph (or use the given flow of a residual network)

        Returns:
            - min_cut_set -- the resulting min cut set of the graph
            - min_cut_value -- the min cut value of the min cut set
            - vertex_sequence -- the sequence of the sink selection
            - residual -- the residual network
            - min_cut_sink -- the sequence of sinks with their calculated min cut set
        """
        # Initialize the minimum cut value and the corresponding cut set
        min_cut_value = float("inf")
        min_cut_set = ({}, {})
        min_cut_sink = []

        # Start with the source node in set S and copy the graph as a residual graph
        S = {source}
        residual = graph.copy()

        # Initialize variables for the modified Hao-Orlin algorithm
        N, W, DormantSet, D_max, t = self.modifiedInitialize(residual, source, sink, reset)
        vertex_sequence = [t]  # Track the sequence of sink nodes

        # Loop until the set S includes all nodes
        while S != N:
            # Process  nodes while there are active nodes in W
            while self.get_active_node(W, residual, t) is not None: #and W != cut_target: #or look_for_certain_cut:
                current_node = self.get_active_node(W, residual, t)

                admissible_arc_node = self.hao_excess_flow_vertex(residual, current_node, W)
                if admissible_arc_node is not None:
                    # Push flow if there's an admissible arc
                    self.hao_push(residual, current_node, W, force=False)
                else:
                    # Relabel if no admissible arc is found
                    W, DormantSet, D_max, residual = self.ModifiedRelabel(current_node, W, DormantSet, residual, D_max)

            # Calculate the cut value after processing the active nodes
            source_list = N - W
            target_list = W.copy()
            cut_value = self.get_cut_value(graph, source_list, target_list)
            if min_cut_value >= cut_value:
                min_cut_value = cut_value
                min_cut_set = (source_list.copy(), target_list.copy())  # Update the minimum cut set

            min_cut_sink.append((t, (source_list, target_list.copy())))

            # If a sink node is specified, break after the first iteration
            if sink is not None:
                break
            else:
                # Select a new sink node and continue if there is no specific sink
                result = self.SelectNewSink(residual, S, W, DormantSet, t, D_max)
                if result is None:
                    break
                S, W, DormantSet, t, D_max = result
                if t is None:
                    break
                vertex_sequence.append(t)  # Track the sequence of sink nodes

        # Return the minimum cut set, its value, and the sequence of vertices processed
        return min_cut_set, min_cut_value, vertex_sequence, residual, min_cut_sink

    def get_active_node(self, W, residual, t):
        """
        Calculates and returns the next active node (if possible)

        Parameters:
            - W -- the active set
            - residual -- the residual graph
            - t -- the sink node

        Returns:
            - node -- if a node is found, it is returned, otherwise None will be returned
        """
        # Iterate over all nodes in W excluding t
        for node in W - {t}:
            # Check if the node has positive excess flow, making it an active node
            if residual.nodes[node]['excess'] > 0:
                return node
        # If no active node is found, return None
        return None

    def get_cut_value(self, graph, D, W):
        """
        Calculates the cut value of the given cut set

        Parameters:
            - graph -- the graph to calculate the min cut
            - D -- the source nodes
            - W -- the sink nodes

        Returns:
            - cut_value -- the value of the cut
        """
        cut_value = 0
        # Iterate over all edges in the graph
        for edge in graph.edges():
            source, target = edge
            # Check if the edge crosses from a node in D to a node in W
            if (source in D) and (target in W):
                # Add the weight (or 'order' attribute) of the edge to the cut value
                cut_value += graph[source][target]['order']
        return cut_value

    def modifiedInitialize(self, residual, source, sink, reset):
        """
        Initalization method of the Hao-Orlin algorithm

        Parameters:
            - residual -- the graph to initialize
            - source -- the source node
            - sink -- the sink node (if not defined, sink will be determined by the algorithm)
            - reset -- defines whether to perform a reset on the graph (or use the given flow of a residual network)

        Returns:
            - N -- the set of all nodes
            - W -- the set of active nodes
            - DormantSet -- the list of dormant node sets
            - D_max -- the maximum distance/height of the dormant set
            - t -- the current sink node
        """
        if reset:
            # Initialize all edges: set their flow to 0
            for u, v in residual.edges():
                residual[u][v]['flow'] = 0

        # Initialize all nodes: set their height to 0 and excess flow to 0
        for node in residual.nodes():
            residual.nodes[node]['height'] = 0
            residual.nodes[node]['excess'] = 0

        # Set the height of the source node to the number of nodes (pushes flow more efficiently)
        residual.nodes[source]['height'] = len(residual.nodes())

        # Initialize the dormant set structure (DormantSet) and maximum height label (D_max)
        DormantSet = [set() for _ in range(len(residual.nodes()) + 1)]
        DormantSet[0] = {source}
        D_max = 0

        # Initialize sets N (all nodes) and W (working set of nodes excluding the source)
        N = set(residual.nodes())
        W = N.copy()
        W.remove(source)

        # Determine the initial sink node: use the provided sink or pick the first node in W
        if sink is not None:
            t = sink
        else:
            t = list(W)[0]  # Pick the first node in W if sink is not provided
        # Set the height of the sink node to 0
        residual.nodes[t]['height'] = 0

        # Set the height of all nodes (except source and sink) to 1, and push flow from source
        for j in N - {t} - {source}:
            residual.nodes[j]['height'] = 1

        # Push flow from the source to its neighbors and update their excess flow
        for neighbor in residual.neighbors(source):
            capacity = residual[source][neighbor]['order']
            if capacity > 0:
                flow = residual[source][neighbor]['flow']
                diff = capacity - flow
                residual[source][neighbor]['flow'] = capacity
                residual[neighbor][source]['flow'] -= diff
                residual.nodes[neighbor]['excess'] += diff

        return N, W, DormantSet, D_max, t

    def hao_push(self, residual, u, nodes, force): # Attempt to push flow from u to each of its neighbors in the residual graph, push minimum of excess or capacity of edge
        """
        The push method of the Hao-Orlin algorithm

        Parameters:
            - residual -- the graph to use
            - u -- the node of the graph from which the push is executed
            - nodes -- the nodes which are allowed to be pushed to
            - force -- defines whether the push should be forced (ignoring the execess and distance/height)

        Returns:
            - (performs the push on the graph reference)
        """
        for v in residual.neighbors(u):
            # Skip if there is no available capacity or if v is not in the working node set
            if (residual[u][v]['order'] - residual[u][v]['flow'] == 0) or v not in nodes:
                continue

            # Calculate the amount of flow to push (delta)
            if not force:
                delta = min(residual.nodes[u]['excess'], residual[u][v]['order'] - residual[u][v]['flow'])
            else:
                delta = residual[u][v]['order'] - residual[u][v]['flow']

            # Push flow if the height of u is greater than v or if the operation is forced
            if (residual.nodes[u]['height'] > residual.nodes[v]['height']) or force:
                residual[u][v]['flow'] += delta  # Increase flow on edge u->v
                residual[v][u]['flow'] -= delta  # Decrease flow on edge v->u (reverse flow)
                residual.nodes[u]['excess'] -= delta  # Decrease excess at u
                residual.nodes[v]['excess'] += delta  # Increase excess at v

                # If not forced, stop after the first valid push
                if not force:
                    return
        return

    def SelectNewSink(self, residual, S, W, DormantSet, t, D_max):
        """
        Method to select a new sink

        Parameters:
            - residual -- the graph to perform the selection on
            - S -- the set of sinks
            - W -- the set of active nodes
            - DormantSet -- the list of dormant node sets
            - t -- the current sink node
            - D_max -- the maximum distance/height of the dormant set

        Returns:
            - S -- the set of sinks (updated)
            - W -- the set of active nodes (updated)
            - DormantSet -- the list of dormant node sets
            - t -- the current sink node (updated)
            - D_max -- the maximum distance/height of the dormant set (updated)
            - (performs the push on the graph reference)
        """
        # Move the current sink t from W to S and add it to DormantSet[0]
        W.remove(t)
        S.add(t)
        DormantSet[0].add(t)

        # If all nodes are in S, return None to signal completion
        if S == set(residual.nodes()):
            return None

        # Force push flow from the current sink t to the remaining nodes in the residual graph
        self.hao_push(residual, t, residual.nodes() - S, force=True)

        # If W is empty, repopulate it with nodes from DormantSet[D_max] and decrease D_max
        if len(W) == 0:
            W = DormantSet[D_max].copy() # Copy the set to avoid modifying DormantSet directly
            D_max -= 1

        # Initialize variables to find the new sink node t with the smallest height
        t = None
        t_dist = float("inf")

        # Find the node in W with the smallest height to be the new sink
        for node in W:
            node_dist = residual.nodes[node]['height']
            if node_dist <= t_dist:
                t = node
                t_dist = node_dist

        return S, W, DormantSet, t, D_max

    def ModifiedRelabel(self,i, W, DormantSet, residual, D_max):
        """
        Relabel method of the Hao-Orlin algorithm

        Cases:
            Case 1: Check if node i has a unique height among nodes in W
            Case 2: Check if there are no edges from node i to any node in W
            Case 3: Standard relabel operation - increase the height of node i

        Parameters:
            - i -- the current node being processed
            - W -- the set of active nodes
            - DormantSet -- the list of dormant node sets
            - residual -- the graph to perform the relabel on
            - D_max -- the maximum distance/height of the dormant set

        Returns:
            - W -- the set of active nodes (updated)
            - DormantSet -- the list of dormant node sets (updated)
            - D_max -- the maximum distance/height of the dormant set (updated)
            - residual -- the graph to perform the relabel on (updated)
        """
        if self.unique_height(i, W, residual):
            D_max += 1  # Increment the maximum height label
            # R: set of nodes in W with height >= height of node i
            R = {j for j in W if residual.nodes[j]['height'] >= residual.nodes[i]['height']}
            DormantSet[D_max] = R.copy()  # Move nodes with height >= i's height to a new dormant set
            W -= R  # Remove these nodes from W
        elif self.no_W_D_edge(i, W, residual):
            D_max += 1
            DormantSet[D_max] = {i}  # Add node i to DormantSet at the new height
            W.remove(i)
        else:
            min_height = float("inf")
            # Find the minimum height among neighbors in W with available capacity
            for neighbor in residual.neighbors(i):
                if neighbor in W and residual[i][neighbor]['order'] - residual[i][neighbor]['flow'] > 0:
                    # Update min_height to be the minimum height found among neighbors
                    min_height = min(residual.nodes[neighbor]['height'] + 1, min_height)
            residual.nodes[i]['height'] = min_height

        return W, DormantSet, D_max, residual

    def no_W_D_edge(self,i, W, residual):
        """
        Check if node i has any outgoing edges to nodes in the set W

        Parameters:
            - i -- the current node being processed
            - W -- the set of active nodes
            - residual -- the graph to use

        Returns:
            - Bool -- whether check was successful
        """
        for j in residual.neighbors(i):
            if j in W and residual[i][j]["order"] - residual[i][j]["flow"] > 0: # residual[i][j]["order"] > 0:
                return False
        return True

    def unique_height(self, i, W, residual):
        """
        Check if i's height is unique in W

        Parameters:
            - i -- the current node being processed
            - W -- the set of active nodes
            - residual -- the graph to use

        Returns:
            - Bool -- whether check was successful
        """
        return all(residual.nodes[j]['height'] != residual.nodes[i]['height'] for j in W if j != i)

    def hao_excess_flow_vertex(self, residual, node, W):
        """
        Checks if a pushable node is present and returns it

        Parameters:
            - residual -- the graph to use
            - node -- the node from which will be pushed
            - W -- the set of active nodes

        Returns:
            - node -- if a node is found, it is returned, otherwise None will be returned
        """
        for neighbor in residual.neighbors(node):  # Iterate through neighbors of the node
            # Check conditions: node has excess flow, neighbor's height is less, neighbor is in W, and there's available capacity
            if (residual.nodes[node]['excess'] > 0) and (
                    residual.nodes[neighbor]['height'] < residual.nodes[node]['height']) and (neighbor in W) and (
                    residual[node][neighbor]['order'] - residual[node][neighbor]['flow'] > 0):
                return node  # Return the node if all conditions are met
        return None  # Return None if no such neighbor is found

    def get_binary(self, nodes_source, nodes_target, k):  # k = total number of nodes
        """
        Calculates and returns the list representation of the cut

        Explanation:
            - list length = total number of nodes
            - position in the list = node number
            - '0' = source node
            - '1' = target node
            - '2' = not yet defined

        Parameters:
            - nodes_source -- the source nodes
            - nodes_target -- the target nodes
            - k -- the total number of nodes

        Returns:
            - bin -- the list representation of the cut
        """
        bin = []
        for i in range(0, k):
            if i in nodes_source:
                bin.append(1)  # Node belongs to source set
            elif i in nodes_target:
                bin.append(0)  # Node belongs to target set
            else:
                bin.append(2)  # Node belongs to neither set
        return bin

    def to_edge_set(self, nodecutbits, G):

        min_cut_source_list, min_cut_target_list = self.get_indizes(nodecutbits)
        base_vector = np.zeros(len(G.edges()), dtype=int) 

        for n in nx.edge_boundary(G, min_cut_source_list):
            base_vector[G.edges[n]['index']] = 1

        return base_vector

    def enumerate_min_cuts(self, basegraph, graph, show_all):
        """
        Main function to enumerate the min cuts

        Parameters:
            - graph -- the graph to use
            - show_all -- defines whether to show all cuts or just cuts which split the graph into 2 parts

        Returns:
            - cut_solutions -- a sorted list of the enumerated cuts (contents of the vector defined below)
                - 0 -- min cut value
                - 1 -- counter to sort cuts with equal min cut value
                - 2 -- list representation of the parent (partially defined) cut
                - 3 -- list representation of the min cut
                - 4 -- the min cut set in the tuple(set, set) representation
                - 5 -- the child node (differing node of the previous cut)
                - 6 -- boolean whether the parent (partially defined) cut set was created in the basic partition
        """
        cut_solutions = []
        cut_vectors = []

        cut_problems = []
        heapq.heapify(cut_problems)  # Initialize heap for cut problems

        self.entry_counter = count()

        for u, v in graph.edges():
            graph[u][v]['flow'] = 0

        # Create the initial cut and determine the target size
        source = 0
        initial_min_cut_set, initial_min_cut_value, initial_sink_vertex_list, residual, _ = self.hao_orlin(graph, source,
                                                                                                        None, True)
        initial_sink_vertex_list.insert(0, 0)

        # Initialize the first cut problem
        initial_cut_problem = (
            initial_min_cut_value, next(self.entry_counter),
            self.get_binary(set([0]), set(), total_number_of_nodes),
            self.get_binary(initial_min_cut_set[0], initial_min_cut_set[1], total_number_of_nodes),
            initial_min_cut_set,
            0,
            True
        )
        cut_problems.append(initial_cut_problem)

        # Loop until all cut problems are processed
        while len(cut_problems) != 0:
            cut = heapq.heappop(cut_problems)  # Get the current cut with the smallest cut value

            # Test for linear independence
            cut_vectors.append( self.to_edge_set(cut[3], basegraph) )  # Binary representation of the cut

            zero_in_cut_vectors_test = False
            elim_cut_vectors_test = gf2elim(np.asarray(cut_vectors))
            for line in elim_cut_vectors_test:
                zero_in_cut_vectors_test |= not any(line)

            if zero_in_cut_vectors_test:
                cut_vectors.pop()
            else:
                cut_solutions.append(cut)

            if len(cut_solutions) == total_number_of_nodes - 1:  # Stop if all cuts are found
                break

            parent = cut[2]

            child_node = cut[5]

            if cut[1] != 0:
                parent_contracted_graph, parent_main_node_source, parent_main_node_target = self.contract_nodes(graph, parent)

                _, _, residual = self.pushrelabel(parent_contracted_graph, parent_main_node_source, parent_main_node_target)

            # Generate child cuts
            if (cut[1]) == 0:
                # calculate children (basic partition)
                childs = self.get_immediate_children(sink_vertex_list=initial_sink_vertex_list)

                # for each calculated children
                for child_triple in childs:
                    child = child_triple[0]
                    child_node = child_triple[2]
                    source_list, target_list = self.get_indizes(child)

                    # calculate the min cut
                    child_contracted_graph, child_main_node_source, child_main_node_target = self.contract_nodes(graph, child)

                    child_min_cut_set, child_min_cut_value, _, _, _ = self.hao_orlin(child_contracted_graph,
                                                                                                child_main_node_source,
                                                                                                child_main_node_target, True)

                    min_cut_set = (child_min_cut_set[0].union(set(source_list)), child_min_cut_set[1].union(set(target_list)))

                    # append the cut to the list of cut problems
                    if show_all or self.is_valid_cut(graph, min_cut_set[0], min_cut_set[1]):
                        new_cut_problem = (
                            child_min_cut_value,
                            next(self.entry_counter),
                            child,
                            self.get_binary(min_cut_set[0], min_cut_set[1], total_number_of_nodes),
                            min_cut_set,
                            child_node,
                            True
                        )
                        heapq.heappush(cut_problems, new_cut_problem)

            else:
                # calculate the residual network
                for u, v, d in residual.edges.data():
                    d["order"] = d["order"] - d["flow"]
                    d["flow"] = 0

                # phase 1
                min_cut_source_list, min_cut_target_list = self.get_indizes(cut[3])
                source_list, target_list = self.get_indizes(parent)

                phase1_graph = residual.copy()

                # remove target nodes from residual graph
                phase1_graph.remove_nodes_from(min_cut_target_list)

                if len(phase1_graph.nodes()) > 1:

                    _, _, _, _, min_cut_sink = self.hao_orlin(phase1_graph,
                                                                parent_main_node_source,
                                                                None, False)

                    # iterate over calculated cuts
                    for sink_cut in min_cut_sink:
                        source_list_2 = source_list.copy()
                        target_list_2 = target_list.copy()

                        # cut of the current sink
                        min_cut_set_sink = sink_cut[1]
                        min_cut_set = (min_cut_set_sink[0].union(set(source_list)), min_cut_set_sink[1].union(set(target_list)).union(set(min_cut_target_list)))
                        cut_value = self.get_cut_value(graph, min_cut_set[0], min_cut_set[1])

                        # calculate the partially specified cut
                        for i in min_cut_sink:
                            if i != sink_cut:
                                source_list_2.append(i[0])
                            else:
                                target_list_2.append(i[0])
                                break

                        child = self.get_binary(source_list_2, target_list_2, total_number_of_nodes)

                        # append the cut to the list of cut problems
                        if show_all or self.is_valid_cut(graph, min_cut_set[0], min_cut_set[1]):
                            new_cut_problem = (
                                cut_value,
                                next(self.entry_counter),
                                child,
                                self.get_binary(min_cut_set[0], min_cut_set[1], total_number_of_nodes),
                                min_cut_set,
                                i[0],
                                False
                            )
                            heapq.heappush(cut_problems, new_cut_problem)

                # phase2
                source_list, target_list = self.get_indizes(parent)

                phase2_graph = residual.copy()

                # remove source nodes from residual graph
                phase2_graph.remove_nodes_from(min_cut_source_list)

                # reverse the edges
                phase2_graph = phase2_graph.reverse()

                if len(phase2_graph.nodes()) == 1:
                    continue

                _, _, _, _, min_cut_sink = self.hao_orlin(phase2_graph,
                                                            parent_main_node_target,
                                                            None, False)

                # iterate over calculated cuts
                for sink_cut in min_cut_sink:
                    source_list_2 = min_cut_source_list.copy()
                    target_list_2 = target_list.copy()

                    # cut of the current sink
                    min_cut_set_sink = sink_cut[1]
                    min_cut_set = (min_cut_set_sink[1].union(set(min_cut_source_list)), min_cut_set_sink[0].union(set(target_list)))
                    cut_value = self.get_cut_value(graph, min_cut_set[0], min_cut_set[1])

                    # calculate the partially specified cut
                    for i in min_cut_sink:
                        if i != sink_cut:
                            target_list_2.append(i[0])
                        else:
                            source_list_2.append(i[0])
                            break

                    child = self.get_binary(source_list_2, target_list_2, total_number_of_nodes)

                    # append the cut to the list of cut problems
                    if show_all or self.is_valid_cut(graph, min_cut_set[0], min_cut_set[1]):
                        new_cut_problem = (
                            cut_value,
                            next(self.entry_counter),
                            child,
                            self.get_binary(min_cut_set[0], min_cut_set[1], total_number_of_nodes),
                            min_cut_set,
                            i[0],
                            False
                        )
                        heapq.heappush(cut_problems, new_cut_problem)
        return cut_solutions

    def get_immediate_children(self, sink_vertex_list):
        """
        Creates the children for the basic partition

        Parameters:
            - sink_vertex_list -- the sequence of the sink selection

        Returns:
            - children -- the children generated by the basic parition
        """
        children = []
        # For basic case, generate children by splitting the sink vertex list incrementally
        source_vertex_list = [0]  # Start with source node as 0
        for i in range(0, len(sink_vertex_list) - 1):
            # Append the current sink to the source list
            source_vertex_list.append(sink_vertex_list[i])
            # Create a new child with the updated source and sink lists
            children.append((
                self.get_binary(set(source_vertex_list), set([sink_vertex_list[i + 1]]), total_number_of_nodes),
                True,  # Indicates it's a source side partition
                sink_vertex_list[i + 1]  # The next sink to consider
            ))
        return children

    def contract_nodes(self, graph, child):
        """
        Contracts the nodes according to the given partially defined cut

        Parameters:
            - graph -- the graph to use
            - child -- the partially defined cut to process

        Returns:
            - modified_graph -- the modified graph with the contracted nodes
            - main_node_source -- the resulting source node (node in which all source nodes are contracted)
            - main_node_target -- the resulting target node (node in which all target nodes are contracted)
        """
        # Get the lists of indices for source and target nodes based on the binary representation (child)
        source_index_list, target_index_list = self.get_indizes(child)

        main_node_source = source_index_list[0]
        main_node_target = target_index_list[0]

        modified_graph = graph.copy()

        # Contract nodes in the source list into main_node_source
        for i in range(1, len(source_index_list)):
            # Skip nodes that are not in the graph
            if not graph.has_node(source_index_list[i]):
                continue

            # Get edge values before contracting nodes
            source_edge_values = self.get_edge_values(modified_graph, [main_node_source, source_index_list[i]])

            # Contract the nodes (combine them into main_node_source)
            modified_graph = nx.contracted_nodes(modified_graph, main_node_source, source_index_list[i],
                                                    self_loops=False)

            # Update the 'order' attribute for outgoing edges of the contracted node
            for u, v, d in modified_graph.out_edges(main_node_source, data=True):
                d["order"] = source_edge_values[v][1]
                d["flow"] = source_edge_values[v][2]

            # Update the 'order' attribute for incoming edges of the contracted node
            for u, v, d in modified_graph.in_edges(main_node_source, data=True):
                d["order"] = source_edge_values[u][1]
                d["flow"] = source_edge_values[u][2]

        # Contract nodes in the target list into main_node_target
        for i in range(1, len(target_index_list)):
            # Skip nodes that are not in the graph
            if not graph.has_node(target_index_list[i]):
                continue

            # Get edge values before contracting nodes
            target_edge_values = self.get_edge_values(modified_graph, [main_node_target, target_index_list[i]])

            # Contract the nodes (combine them into main_node_target)
            modified_graph = nx.contracted_nodes(modified_graph, main_node_target, target_index_list[i],
                                                    self_loops=False)

            # Update the 'order' attribute for outgoing edges of the contracted node
            for u, v, d in modified_graph.out_edges(main_node_target, data=True):
                d["order"] = target_edge_values[v][1]
                d["flow"] = target_edge_values[v][2]

            # Update the 'order' attribute for incoming edges of the contracted node
            for u, v, d in modified_graph.in_edges(main_node_target, data=True):
                d["order"] = target_edge_values[u][1]
                d["flow"] = target_edge_values[u][2]

        return modified_graph, main_node_source, main_node_target

    def get_edge_values(self, graph, node_list):
        """
        Function to sum the edge values of the given nodes

        Parameters:
            - graph -- the graph to use
            - node_list -- the list of nodes to be contracted

        Returns:
            - edge_values -- list of triples which represent the resulting edge values to the nodes (triple defined below)
                - 0 -- neighboring node
                - 1 -- resulting 'order' value for the edge
                - 2 -- resulting 'flow' value for the edge
        """
        global total_number_of_nodes

        # Initialize edge_values with a tuple (node, 0.0) for each node in the graph
        edge_values = []
        for node in range(total_number_of_nodes):
            edge_values.append((node, float(0), float(0)))

        # Update edge_values for each node in node_list
        for node in node_list:
            for u, v, d in graph.out_edges(node, data=True):
                # For each outgoing edge from 'node', update the 'order' value in edge_values for the target node 'v'
                edge_values[v] = (edge_values[v][0], edge_values[v][1] + d['order'], edge_values[v][2] + d['flow'])

        return edge_values

    def get_indizes(self, bits):
        """
        Calculates the seperated source and target list from the list representation

        Parameters:
            - bits -- list representation of the cut

        Returns:
            - source_index_list -- list of the source nodes
            - target_index_list -- list of the target nodes
        """
        source_index_list = []
        target_index_list = []
        index = 0

        for i in bits: # Iterate over each bit in  'bits'
            if i == 1:
                source_index_list.append(index)  # Add index to source list if bit is 1
            elif i == 0:
                target_index_list.append(index)  # Add index to target list if bit is 0
            index += 1  # Increment index for the next bit

        return source_index_list, target_index_list

    def is_valid_cut(self, graph, source_index_list, target_index_list):
        """
        Checks whether the cut is 'valid' (cuts which split the graph into 2 parts)

        Parameters:
            - graph -- the graph to use
            - source_index_list -- list of the source nodes
            - target_index_list -- list of the target nodes

        Returns:
            - boolean whether the check was successful
        """
        if not source_index_list or not target_index_list:
            return False  # Return False if either list is empty

        # Check if the source subgraph is connected
        if not source_index_list:
            source_connected = True  # Consider empty source as connected
        else:
            source_subgraph = graph.subgraph(source_index_list)  # Extract the subgraph for source indices
            source_connected = nx.is_weakly_connected(source_subgraph)  # Check if the subgraph is weakly connected

        # Check if the target subgraph is connected
        if not target_index_list:
            target_connected = True  # Consider empty target as connected
        else:
            target_subgraph = graph.subgraph(target_index_list)  # Extract the subgraph for target indices
            target_connected = nx.is_weakly_connected(target_subgraph)  # Check if the subgraph is weakly connected

        # Return True only if both subgraphs are connected
        return source_connected and target_connected


    def pushrelabel(self, graph, source, sink):
        # Initialize a copy of the graph and initialize flow from the source
        residual = graph.copy()
        self.preflow(residual, source)
        # Find a node with excess flow that can push flow to a neighbor
        node = self.excess_flow_vertex(residual, sink)
        while(node != None): # While there exists a node with excess flow
            # Try to push flow from the node; if push isn't possible, but node still has excess flow, relabel the node
            if not (self.push(residual, node)):
                self.relabel(residual, node)
            # Update the node to the next vertex with excess flow
            node = self.excess_flow_vertex(residual, sink)

        # Calculate the max flow by checking the excess flow at the sink
        max_flow = residual.nodes[sink]['excess']
        # Create a copy of the residual graph and remove edges with no residual capacity
        residual_copy = residual.copy()
        for (u, v) in residual.edges():
            if not residual[u][v]['order'] - residual[u][v]['flow'] > 0:
                residual_copy.remove_edge(u, v)

        # Determine source/sink partitions of the graph by checking their reachability from the source in the residual graph
        source_partition = {source}
        sink_partition = {sink}
        for node in residual_copy.nodes():
            if nx.has_path(residual_copy, source, node):
                source_partition.add(node)
            else:
                sink_partition.add(node)
        
        #output max flow (min cut_value) and we also need partition of min cut
        return max_flow, (source_partition, sink_partition), residual

    def preflow(self, residual, source):
        # Initialize the height and excess flow for all nodes to zero
        for node in residual.nodes():
            residual.nodes[node]['height'] = 0
            residual.nodes[node]['excess'] = 0
        # Initialize the flow on all edges to zero
        for u, v in residual.edges():
            residual[u][v]['flow'] = 0
        # Set the source node's height to the number of nodes
        residual.nodes[source]['height'] = len(residual.nodes())
        # Push full capacity from the source to all its neighbors and adjust the excess flow from the nodes
        for neighbor in residual.neighbors(source):
            capacity = residual[source][neighbor]['order']
            residual[source][neighbor]['flow'] = capacity
            residual[neighbor][source]['flow'] = -capacity
            residual.nodes[neighbor]['excess'] = capacity
            residual.nodes[source]['excess'] -= capacity

    def push(self, residual, u): # Attempt to push flow from node u to its neighbors
        for v in residual.neighbors(u):
            # Skip if the edge from u to v is fully saturated (no remaining capacity)
            if (residual[u][v]['order'] - residual[u][v]['flow'] == 0):
                continue
            # Calculate the amount of flow to push (minimum of u's excess or remaining capacity on the edge)
            delta = min(residual.nodes[u]['excess'], residual[u][v]['order'] - residual[u][v]['flow'])
            # Push flow if u is higher than v (height condition) and adjust the excess flow of both nodes
            if (residual.nodes[u]['height'] > residual.nodes[v]['height']):
                residual[u][v]['flow'] += delta
                residual[v][u]['flow'] -= delta
                residual.nodes[u]['excess'] -= delta
                residual.nodes[v]['excess'] += delta
                return True # Push operation was successful
        return False # No push was possible

    def relabel(self, residual, u): # Increase the height of node u to enable a future push operation
        min_height = float('inf')
        # Find the minimum height of neighbors where more flow can be pushed
        for v in residual.neighbors(u):
            if residual[u][v]['order'] > residual[u][v]['flow']:
                min_height = min(min_height, residual.nodes[v]['height'])
        # Set the height of u to one more than the minimum neighbor height
        residual.nodes[u]['height'] = min_height + 1

    def excess_flow_vertex(self, residual, sink):
        # Find a node (excluding the sink) with positive excess flow
        for node in residual.nodes():
            if (residual.nodes[node]['excess'] > 0) and (node != sink):
                return node # Return the first node found with excess flow
        return None
    def find_overall_mincut(self, source, graph):
        source_node = source
        # Initialize variables to store the minimum cut set and its value
        min_cut_value = float('inf')
        min_cut_set = set()

        for sink_node in graph.nodes():  # Iterate over all nodes in the graph that are not the source node
            if sink_node != source_node:
                # Compute the minimum cut between the source and the current sink node
                cut_value, partition, _ = self.pushrelabel(graph, source_node, sink_node)
                # Update the minimum cut value and set if necessary
                if cut_value < min_cut_value:
                    min_cut_value = cut_value
                    min_cut_set = partition

        return min_cut_set, min_cut_value
