import matplotlib.pyplot as plt
import networkx as nx
from networkx import algorithms
from networkx.algorithms import flow
from copy import copy
import sys
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


def print_graph(pg, name, att='capacity'):
    #return

    plt.figure()
    pos = nx.spring_layout(pg)  # Position nodes in a visually appealing way
    nx.draw(pg, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_color='black')
    edge_labels = nx.get_edge_attributes(pg, att)
    nx.draw_networkx_edge_labels(pg, pos, edge_labels=edge_labels)
    plt.savefig(name)
    plt.close()

def condensed_print_graph(W, name, att='capacity'):

    dag = nx.condensation(W)
    relabel = { n: ",".join( [str(k) for k,v in dag.graph['mapping'].items() if v == n] ) for n in dag.nodes() }
    nx.relabel_nodes(dag, relabel, copy=False)
             
    tr_dag = nx.transitive_reduction(dag)
    tr_dag.add_nodes_from(dag.nodes(data=True))
    
    print_graph(tr_dag, name, att)

def fast_reduced_transitive_closure_graph(G, s, t):

    DG = G.to_directed()
    flow_value, flow_dict = nx.maximum_flow(G, s, t, capacity='order')

    H = nx.DiGraph()
    for i,j,prop in DG.edges(data=True):
        if 0 <= flow_dict[i][j]  < prop['order']:
            H.add_edge(i,j)
        if flow_dict[i][j]  > 0:
            H.add_edge(j,i)

    U = nx.DiGraph()

    anc = list(nx.ancestors(H, t))
    anc.append(t)
    for a1 in anc:
        for a2 in anc:
            if a1 != a2:
                U.add_edge(a1, a2)

    desc = list(nx.descendants(H, s))
    desc.append(s)
    for d1 in desc:
        for d2 in desc:
            if d1 != d2:
                U.add_edge(d1, d2)

    for n in H.nodes():
        H.nodes[n]['successors'] = list(nx.descendants(H, n))

    for i in G.nodes():
        for j in G.nodes():
            if i == j:
                continue
            if j in H.nodes[i]['successors'] and G.has_edge(i,j):
                U.add_edge(i,j)
    return U
        
def enum_closed_sets(dag):

    for i, n in enumerate(nx.topological_sort(dag)):
        dag.nodes[n]['toporder'] = i

    zero_indegree = set()
    for v, d in dag.in_degree():
        dag.nodes[v]['indeg'] = d
        if d == 0:
            zero_indegree.add(v)

    if len(zero_indegree) > 1:
        print("WARNING: multiple initial zero indegree nodes", file=sys.stderr)
        return None

    raw_cuts = []
    enum_closed_sets_recursion(dag, zero_indegree, 0, raw_cuts)
    
    #expand the condensed nodes to the original node set
    cuts = []
    for rc in raw_cuts:
        cuts.append( [ x for cn in rc for x in dag.nodes[cn]['members'] ] )
    
    return cuts

def enum_closed_sets_recursion(dag, cutset, topo_label, res):

    res.append(cutset)    

    open_nodes = {}
    for e in nx.edge_boundary(dag, cutset): 
 
        n2 = e[0]
        if n2 in cutset:
            n2 = e[1]
        
        open_nodes[n2] = open_nodes.setdefault(n2, 0) + 1

    for n in open_nodes:
        if ( open_nodes[n] == dag.nodes[n]['indeg'] # nodes that have no incoming edges left in with cutset gone
                and dag.nodes[n]['toporder'] > topo_label   # check to only add nodes lower than current topo level to avoid listing nodes twice
                and dag.nodes[n]['toporder'] < len(dag.nodes()) - 1):  # don't ever add the last node

            next_cutset = cutset.copy()
            next_cutset.add(n)
            enum_closed_sets_recursion(dag, next_cutset, dag.nodes[n]['toporder'], res)

def cut_to_binary(index_map, G, cut):
    base_vector = np.zeros(len(G.edges()), dtype=int) 

    for n in nx.edge_boundary(G, cut):
        base_vector[index_map[n]] = 1

    return base_vector

def translate_binary_to_sets(rev_index_map, cut_base, cut_weights):

    cut_sets = []
    for cut, weight in zip(cut_base, cut_weights):

        cut_set= set()
        for i in range(cut.shape[0]): 
            if cut[i] == 1:
                cut_set.add(rev_index_map[i])

        cut_sets.append( (weight, cut_set) )
    return cut_sets

def enumerate_all_minimum_cuts(G):

    index_map = {}
    rev_index_map = {}
    for i, (u,v) in enumerate(G.edges()):
        index_map[(u,v)] = i
        index_map[(v,u)] = i

        rev_index_map[i] = (u,v)

    # create equivalent flow tree 
    T = nx.gomory_hu_tree(G, capacity='order')
    #print_graph(T, "T", att = "weight")
    
    weight_groups = {}
    for u,v, d in T.edges(data=True):
        weight_groups.setdefault(d['weight'],[]).append((u,v))
    
    cut_base = []
    cut_weights = []
    for weight in sorted(weight_groups):
        for u,v in weight_groups[weight]:

            tcg = fast_reduced_transitive_closure_graph(G,u,v)
            dag = nx.condensation(tcg)

            relabel = { n: ",".join( [str(k) for k,v in dag.graph['mapping'].items() if v == n] ) for n in dag.nodes() }
            nx.relabel_nodes(dag, relabel, copy=False)

            cuts = enum_closed_sets(dag)

            for cut in cuts:
                cbin = cut_to_binary(index_map, G, cut)

                cut_base.append(cbin)
                cut_weights.append(weight)

                if len(cut_base) == 1:
                    continue

                zero_in_cut_vectors_test = False
                elim_cut_vectors_test = gf2elim(np.asarray(cut_base))
                for line in elim_cut_vectors_test:
                    zero_in_cut_vectors_test |= not any(line)
            
                if zero_in_cut_vectors_test:
                    cut_base.pop()
                    cut_weights.pop()
            
                if len(cut_base) == len(G.nodes()) - 1:  # Stop if all cuts are found
                    return translate_binary_to_sets(rev_index_map, cut_base, cut_weights)

    return None      

def edge_cut_set_to_partition(G, cut_set):
    
    GC = G.copy()
    for cut in cut_set:
        GC.remove_edge(*cut)

    return list(nx.connected_components(GC))
    
G = nx.read_graphml(sys.argv[1])
# Change node labels to integers

label_to_int = {label: i for i, label in enumerate(G.nodes())}
nx.relabel_nodes(G, label_to_int, copy=False)

#print_graph(G, "Base")
cut_sets = enumerate_all_minimum_cuts(G)

print("============ Final Cut Base ============")      

for weight, cs in cut_sets:    
    print(weight, sorted(cs), edge_cut_set_to_partition(G, cs))

