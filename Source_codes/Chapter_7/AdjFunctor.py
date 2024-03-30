import dgl
import collections
from tqdm.auto import tqdm
import torch

"""The simplicial set needs to be a dictionary with all simplices being lists, including 0-simplices. These should all be put into a giant list itself, and will be the keys of the dicitionary. The values of the dictionary are the dimensions of the simplices. For example, the non-degenerate simplices of the three simplex may be written as {0:[[0],[1],[2],[3]],1:[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],2:[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],3:[[0,1,2,3]]}

The output is a graph without multiedges. In this case, our input graph is homogeneous."""

class AdjFunctor():
    def __init__(self,simplicial_set):
        self.simplicial_set        = simplicial_set
        self.adj_graph_node_dict   = dict()
        self.adj_graph_simplex_ids = dict()
        self.node_index            = 0
        for list_of_simplices in self.simplicial_set.values():
            for simplex in list_of_simplices:
                if simplex == []:
                    continue
                self.adj_graph_node_dict.update({self.node_index: simplex})
                self.adj_graph_simplex_ids.update({id(simplex):self.node_index})
                self.node_index = self.node_index + 1
        self.dimensions = list(self.simplicial_set.keys())
        self.max_dim         = max(self.dimensions)
        self.adj_graph = dgl.graph(([],[]))

            

    def MappingPossibility(self,possiblesublist, possiblesuperlist):
        """
        checks if one list is a sublist of the other by simply
        counting the number of entries
        """
        if possiblesublist == [] or possiblesuperlist == []:
            return False
        c1 = collections.Counter(possiblesublist)
        c2 = collections.Counter(possiblesuperlist)
        for item, counted in c1.items():
            if counted > c2[item]:
               return False
        return True

    def fill_edges(self):
        print("Adding edges to the adjacency graph dimension-wise..")
        for d in tqdm(range(self.max_dim), position=0, leave=True):
            for u_simplex in tqdm(self.simplicial_set[d+1], position=0, leave=True):
                for l_simplex in self.simplicial_set[d]:
                    if self.MappingPossibility(l_simplex, u_simplex):
                        u_node = self.adj_graph_simplex_ids[id(u_simplex)]
                        l_node = self.adj_graph_simplex_ids[id(l_simplex)]
                        self.adj_graph.add_edges(u_node,l_node)