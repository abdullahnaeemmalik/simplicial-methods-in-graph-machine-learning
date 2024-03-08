#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dgl
import torch
from tqdm import tqdm
import json


# This builds the 1-skeleton of a standard simplex

# In[ ]:


class SimplexCreator():
    """Create standard simplex"""
    def __init__(self, dimension):
        self.input_dimension = dimension
        self.src=list()
        self.dst=list()
        for i in range(self.input_dimension+1):
            for j in range(self.input_dimension+1):
                if (i < j):
                    self.src = self.src + [i]
                    self.dst = self.dst + [j]


# If $A$ is the adjacency matrix, we construct $\tilde{A}$, which replaces all entries in the diagonal with a zero, effectively killing off all loops. For the matrix $A^2$, we use Boolean algebra to save memory by reducing the size of the entries of the matrix. A nonzero entry $a_{ij}$ of $\tilde{A} \circ \tilde{A}^2$ corresponds to 2 vertices $i$ and $j$ of a 2-simplex. For all $x$ in the out-neighborhood of $i$ and the in-neighborhood of $j$, we get a 2-simplex $[i,x,j]$.

# In[ ]:


class simplices_2():
    src=list()
    dst=list()
    empty_graph = dgl.heterograph({('node', 'to', 'node'): (src, dst)})

    assert isinstance(empty_graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of AdjGraph\'s init methodmust be a dgl.DGLHeteroGraph.'

    def __init__(
        self, file_path, graph=empty_graph):
        self.seed_graph = graph
        self.file_path = file_path
        seed_edge_pairs = []
        src, dst = self.seed_graph.edges()
        for i, u in enumerate(src):
            v = dst[i]
            seed_edge_pairs.append((int(u),int(v)))
        self.seed_edge_pairs = seed_edge_pairs
        
        self.simplices = {0: [int(x) for x in self.seed_graph.nodes()], 
                          1: self.seed_edge_pairs, 2: list()}
        
        print("Finished adding 0-simplices and 1-simplices in main dictionary")
        
        print("Computing matrices..")
        loopless = dgl.transforms.RemoveSelfLoop()
        graph = loopless(self.seed_graph)
        adj_squared = torch.sparse.mm(graph.adj_external(),graph.adj_external())
        diagonal_mask = (adj_squared._indices()[0] == adj_squared._indices()[1])
        off_diagonal_mask = ~diagonal_mask
        #set all zero values to one where the edge is not a loop
        adj_squared._values()[off_diagonal_mask] = 1.0
        #create a new sparse matrix with diagonal elements killed off
        new_indices = adj_squared._indices()[:, off_diagonal_mask]
        #only use original nonzero values (which were later changed to 1)
        new_values = adj_squared._values()[off_diagonal_mask]
        new_size = adj_squared.size()
        squared_no_diag_binary = torch.sparse_coo_tensor(indices=new_indices,
                                                         values=new_values, size=new_size)
        #the hadamard product is sparse, but keeps track of entries that are zero
        edges_1 = squared_no_diag_binary._indices().transpose(0, 1)
        edges_2 = graph.adj_external()._indices().transpose(0, 1)
        edges = find_common_tensors(edges_1,edges_2)
        adj_size = len(self.seed_graph.nodes())
        ones = torch.ones(edges.shape[0], dtype=torch.int64)
        self.hadamard_product = torch.sparse_coo_tensor(indices=edges.t(), values=ones,
                                                   size=torch.Size([adj_size, adj_size]))
        
    def out_nodes_as_int(self, vertex):
        """convert successors to a list with integer node values"""
        neighbors = [int(v) for v in list(self.seed_graph.successors(vertex))]
        if int(vertex) in neighbors:
            neighbors.remove(int(vertex))
        return neighbors

    def in_nodes_as_int(self, vertex):
        """convert predecessors to a list with integer node values"""
        neighbors = [int(v) for v in list(self.seed_graph.predecessors(vertex))]
        if int(vertex) in neighbors:
            neighbors.remove(int(vertex))
        return neighbors       
    
    def main_search(self):
        print("Adding 2-vertices to the dictionary")
        row_indices, col_indices = self.hadamard_product._indices()
        for i,j in tqdm(zip(row_indices,col_indices), position=0, leave=False):
            intersection = set.intersection(set(self.out_nodes_as_int(i)),
                                            set(self.in_nodes_as_int(j)))
            for k in intersection:
                self.simplices[2] = self.simplices[2] + [(int(i),int(k),int(j))]
            
        print("Finished adding simplices of dimension 2")    
        
        with open(self.file_path, "w") as file:
            json.dump(self.simplices, file)
        print("Dictionary of simplices saved as a JSON file")
        
def find_common_tensors(tensor_A,tensor_B):
    equal_pairs = torch.all(tensor_A[:, None, :] == tensor_B[None, :, :], dim=2)
    common_pair_indices = torch.nonzero(equal_pairs, as_tuple=False)
    return tensor_A[common_pair_indices[:, 0]]


# In[ ]:


"""Code testing with real world data"""
dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
arxiv_graph = dataset.graph[0]
filepath = 'arxiv_graph_2_simplices'
arxiv_preprocessing = simplices_2(filepath,graph=arxiv_graph)
arxiv_preprocessing.main_search()


# In[ ]:


"""Code testing with standard simplices"""
K_10 = dgl.heterograph({('paper', 'cites', 'paper'): (SimplexCreator(dimension=20).src, SimplexCreator(dimension=20).dst)})
filepath = 'K_10'
K_10_preprocessing = simplices_2(filepath,graph=K_10)
K_10_preprocessing.main_search()
print("Simplices dictionary=",K_10_preprocessing.simplices)


# In[ ]:


"""Code testing"""
bell_graph_src  = [0,0,0,1,1,2] + [1] + [4, 4, 4, 5, 5, 6] 
bell_graph_dst = [1,2,3,2,3,3] + [4] + [5, 6, 7, 6, 7, 7]
bell_graph = dgl.heterograph({('paper', 'cites', 'paper'): (bell_graph_src, bell_graph_dst)})
filepath = 'bell_graph'
bell_graph_preprocessing = simplices_2(filepath,graph=bell_graph)
bell_graph_preprocessing.main_search()
print("Simplices dictionary=", bell_graph_preprocessing.simplices)


# In[ ]:


import random
def generate_random_graph(num_nodes):
    src_edges =[]
    dst_edges = []
    edges = []
    for i in range(2*num_nodes):
        src_edges.append(random.randint(0,num_nodes))
        dst_edges.append(random.randint(0,num_nodes))
        edges.append((src_edges[i],dst_edges[i]))
    graph = dgl.heterograph({('paper', 'cites', 'paper'): (src_edges, dst_edges)})
    return graph, edges


# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx

dgl_G, edges = generate_random_graph(10)
print(edges)
nx_G = nx.DiGraph()
nx_G.add_edges_from(edges)
options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}
#pos = nx.spring_layout(nx_G, seed=42)
pos = nx.planar_layout(nx_G)
nx.draw_networkx(nx_G, pos, with_labels=True, node_color='lightblue', node_size=200, font_size=10, font_color='black', arrows=True)
plt.show()


# In[ ]:


filepath='randomgraph'
random_graph_preprocessing = simplices_2(filepath,graph=dgl_G)
random_graph_preprocessing.main_search()
print("Simplices dictionary=",random_graph_preprocessing.simplices)

