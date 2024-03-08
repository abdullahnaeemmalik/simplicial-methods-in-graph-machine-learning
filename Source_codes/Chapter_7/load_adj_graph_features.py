import dgl
import json
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.utils import save_graphs
import random


class CreateFeatures():
    """If we have node as a simplex [v_0,...v_n] and if the maximum dimension of the simplicial set is k, then for each node of the adjacency graph, we get the vector f(v_0) + f(v_1) + ... + f(v_n) + 0. Here, f(v_i) is the feature vector of node v_i of the original graph, and '+' represents direct sum. If f(v_i) has dimension d, then the zero vector has dimension k(n+1)-d(n+1)
    Returns the graph with such feature vectors"""
    
    
    def __init__(self,file_path_for_adj_graph, file_path_for_mapping, original_graph,features_added_path):
        
        if not "feat" in original_graph.ndata:
            print("Error! The graph should have node features called 'feat'.")
            return
        else:
            self.existing_node_features = original_graph.ndata['feat']
            self.dimension_of_feature_vectors = len(self.existing_node_features[0])

        arxiv_adj_graph, label_dict = dgl.load_graphs(file_path_for_adj_graph)
        self.arxiv_adj_graph = arxiv_adj_graph[0]    
        
        with open(file_path_for_mapping, "r") as file:
            loaded_data = json.load(file)
        
        self.adj_graph_node_dict = {int(key): list(value) for key, value in loaded_data["adj_graph_node_dict"].items()}
        
        dimensions = []
        
        for simplex in self.adj_graph_node_dict.values():
            dimensions = dimensions + [len(simplex) - 1]
            
        self.maximum_dimension = max(dimensions,default=0)
        concatenated_features = []
        all_time_stamps = []
        
        for simplex_node_id in self.arxiv_adj_graph.nodes():
            simplex_node = self.adj_graph_node_dict[int(simplex_node_id)]
            length = len(simplex_node)
            concatenated_feature = torch.zeros(0)
            loc_time_stamps = []
            
            for node in simplex_node:
                concatenated_feature = torch.cat((concatenated_feature,self.existing_node_features[node],), dim=0)
                loc_time_stamps = loc_time_stamps + [int(original_graph.ndata['year'][node])]
                
            remainder = self.maximum_dimension - length + 1
            
            for _ in range(remainder):
                concatenated_feature = torch.cat((concatenated_feature,torch.zeros(self.dimension_of_feature_vectors)), dim=0)
                
            concatenated_features.append(concatenated_feature)
            all_time_stamps.append([random.choice(loc_time_stamps)])

            
        concatenated_features_non_deg = torch.stack(concatenated_features, dim=0)
        self.arxiv_adj_graph.ndata['feat'] = concatenated_features_non_deg
        self.arxiv_adj_graph.ndata['year'] = torch.tensor(all_time_stamps)
        save_graphs(features_added_path, self.arxiv_adj_graph)
    
"""code testing"""
src = [0,0,1]
dst = [1,2,2]
two_simplex = dgl.graph((src,dst))
two_simplex.ndata['feat'] = torch.rand(3,128)
two_simplex.ndata['year'] = torch.tensor([[2017],[2018],[2019]])

file_path_for_mapping = "./arxiv_graph_mapping_dict"
file_path_for_graph = "./arxiv_graph"
features_added_path = "./arxiv_adj_graph_with_features"
adj = CreateFeatures(file_path_for_graph,file_path_for_mapping,two_simplex,features_added_path)
#print(two_simplex.ndata)
#print(adj.arxiv_adj_graph.ndata['year'])
#print(adj.adj_graph_node_dict)
