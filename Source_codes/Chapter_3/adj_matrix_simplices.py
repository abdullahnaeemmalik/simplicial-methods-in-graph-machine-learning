# # Determining 1-skeleton for potential $k$-simplices using adjacency matrix of a graph $G$

# We initialize our graph as a DGL, heterograph. As an example, we use the graph with 8 vertices and 2 3-simplices, connected by a 1-simplex.

# In[1]:


import dgl
import itertools
import torch
from itertools import product


# One of the inputs requires setting the size the highest dimension $d$ to be searched for. If, however, there are no potential $k$-simplices to be found, for $k<d$, the algorithm automatically terminates.

# In[10]:


class GraphBulletPaths():
    src=list()
    dst=list()
    empty_graph = dgl.heterograph({('node', 'to', 'node'): (src, dst)})
    upper_dimension = int
    
    assert isinstance(empty_graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of AdjGraph\'s init methodmust be a dgl.DGLHeteroGraph.'
    
    def __init__(
        self, graph=empty_graph, 
        dimension=upper_dimension):
        self.seed_graph = graph
        src, dst = self.seed_graph.edges()
        src = src.tolist()
        dst = dst.tolist()
        self.dimension = dimension
        set_of_nodes = set(src+dst)
        seed_nodes = list(set_of_nodes)
        self.seed_nodes= [[p] for p in seed_nodes]
        edge_pairs=list(map(list, zip(src, dst)))
        
        """
        These dictionaries are keyed with a dimension, and the corresponding values are vertices of the
        graph that determine a simplex. 
        """
        
        self.simplex_dictionary  = dict()
        
        for i in range(dimension+1):
            self.simplex_dictionary[i]  = list()
            
        self.simplex_dictionary[0]=self.seed_nodes
        self.simplex_dictionary[1]=edge_pairs

        """Most of the calculations below follow rely on the manipulation of an adjacency matrix of the 
        graph, given as a coordinate matrix A=adj_matrix_coo. Powers of each adjacency matrix are stored in
        another list.
        """
        
        A=self.seed_graph.adj(scipy_fmt='coo')
        self.adj_matrix_coo=A
        B=list()
        B.append(list())
        B[0]=A
        # A list with k-th entry corresponding to the power A^{k+1}
        self.powers_of_adj=B
        
    def iterating_through(self):

        # Run through all powers of A^k
        for k in range(2,self.dimension+1):
            self.powers_of_adj.append(self.powers_of_adj[k-2].dot(self.adj_matrix_coo))
            # Record coordinates of the coordinate matrix corresponding to A^{k+1}
            row_indices, column_indices = self.powers_of_adj[k-1].nonzero()
            # If there is no vertex from which originates a path of length k, then we need to stop our search
            if len(row_indices) == 0:
                return k
            
            """The first thing we do is look for a pair of vertices, corresponding to entries in A^{k+1}, labelled
            row_next and col_next. Once these are found, then we look at the same entry for A^k. If this same entry
            is also nonzero, then we record this pair.
            """
        
            for row_next, col_next in zip(*self.powers_of_adj[k-1].nonzero()):
            
                """Temporary lists to capture vertices."""
            
                out_simplices=list()
                in_simplices=list()
                out_neighbors=list()
                in_neighbors=list()
                temp = self.simplex_dictionary[k]
                
                for row_prev, col_prev in zip(*self.powers_of_adj[k-2].nonzero()):
                    if row_prev == row_next:
                        out_simplices.append(col_prev)
                    if col_prev == col_next:
                        in_simplices.append(row_prev)
                
                if k == 2:
                    beta = list(set(in_simplices).intersection(set(out_simplices)))
                    for index in beta:
                        if self.powers_of_adj[0].tocsr()[row_next,col_next] != 0:
                            temp.append([row_next] + [index] + [col_next])
                    continue
                                    
                for edge_r_idx, edge_c_idx in zip(*self.powers_of_adj[0].nonzero()):
                    if edge_r_idx == row_next:
                        out_neighbors.append(edge_c_idx)
                    if edge_c_idx == col_next:
                        in_neighbors.append(edge_r_idx)
                
                beta_one = list(set(in_simplices).intersection(set(out_neighbors)))
                beta_two = list(set(out_simplices).intersection(set(in_neighbors)))
                                
                for index_one, index_two in product(beta_one, beta_two):
                    if self.powers_of_adj[0].tocsr()[index_one, index_two] != 0:
                        if self.powers_of_adj[0].tocsr()[row_next,index_two] != 0:
                            if self.powers_of_adj[0].tocsr()[index_one,col_next] != 0:
                                if self.powers_of_adj[0].tocsr()[row_next,col_next] != 0:
                                    if k == 3:
                                        temp.append([row_next] + [index_one] + [index_two] + [col_next])
                                    if k > 3:
                                        indices_set=set()
                                        for simplex in self.simplex_dictionary[k-1]:
                                            if simplex[-1] == col_next:
                                                if simplex[-2] == index_two:
                                                    if (simplex[0] == row_next or (simplex[0] == index_one)):
                                                        if simplex == ([row_next] + [index_one] + [index_two] + [col_next]):
                                                            continue
                                                        indices_temp_one=simplex.copy()
                                                        indices_temp_one.pop(0)
                                                        indices_temp_one.pop(-1)
                                                        indices_temp_one.pop(-1)
                                                        indices_set = indices_set.union(set(indices_temp_one))
                                            if simplex[0] == row_next:
                                                if simplex[1] == index_one:
                                                    if simplex[-1] == col_next or (simplex[-1] == index_two):
                                                        if simplex == ([row_next] + [index_one] + [index_two] + [col_next]):
                                                            continue
                                                        indices_temp_two=simplex.copy()
                                                        indices_temp_two.pop(0)
                                                        indices_temp_two.pop(0)
                                                        indices_temp_two.pop(-1)
                                                        indices_set = indices_set.union(set(indices_temp_two))
                                        lookupindex= k-3
                                        for idx in itertools.product(indices_set, repeat=lookupindex):
                                            idx = list(idx)
                                            if idx in self.simplex_dictionary[k-4]:
                                                if self.powers_of_adj[0].tocsr()[idx[-1],index_two] != 0:
                                                    if self.powers_of_adj[0].tocsr()[index_one,idx[0]] != 0:
                                                        temp.append([row_next] + [index_one] + list(idx) + [index_two] + [col_next])
                                        self.simplex_dictionary[k] = temp