# -*- coding: utf-8 -*-
from AdjFunctor import AdjFunctor
import json
from dgl.data.utils import save_graphs

with open('arxiv_graph_2_simplices', "r") as file:
    arxiv_graph_2_simplices = json.load(file)
    
        
arxiv_graph_2_simplices_dict = {int(key): list(value)
                                for key, value in arxiv_graph_2_simplices.items()}

#arxiv_graph_2_simplices_dict = {0:[0,1,2],1:[[0,1],[0,2],[1,2]],2:[[0,1,2]]}

zero_simplices = arxiv_graph_2_simplices_dict[0]

zero_simplices_isolated = list()

for zero_simplex in zero_simplices:
    zero_simplices_isolated = zero_simplices_isolated + [[zero_simplex]]
    
arxiv_graph_2_simplices_dict[0] = zero_simplices_isolated

                                
adj_graph = AdjFunctor(arxiv_graph_2_simplices_dict,deg_simplices=False)
adj_graph.fill_edges()

save_graphs("./arxiv_graph", adj_graph.adj_graph)

print("saving mapping dictionary..")
data_to_save = {"adj_graph_node_dict": {str(key): value for key, value in adj_graph.adj_graph_node_dict.items()}, "adj_graph_simplex_ids": {str(key): value for key, value in adj_graph.adj_graph_simplex_ids.items()}}

with open("./arxiv_graph_mapping_dict", "w") as file:
    json.dump(data_to_save, file)
print(adj_graph.adj_graph_node_dict)
print(adj_graph.adj_graph_simplex_ids)
