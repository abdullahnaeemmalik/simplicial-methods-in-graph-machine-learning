import dgl
import json
import torch
from dgl.data.utils import save_graphs


class CreateFeatures():
    """If we have node as a simplex [v_0,...v_n] and if the maximum dimension of the simplicial set is k, then for each node of the adjacency graph, we get the vector f(v_0) + f(v_1) + ... + f(v_n) + 0. Here, f(v_i) is the feature vector of node v_i of the original graph, and '+' represents direct sum. If f(v_i) has dimension d, then the zero vector has dimension k(n+1)-d(n+1)
    Returns the graph with such feature vectors
    mapping is a dictionary where keys are node numbers and values are simplices
    """
    
    
    def __init__(self, adj_graph, mapping, original_graph, tosave:str, data_leak = False):
        self.existing_node_features = original_graph.ndata['feat']
        training_nodes = torch.nonzero(original_graph.ndata['train_mask']).flatten()
        self.dimension_of_feature_vectors = len(self.existing_node_features[0])
            
        self.orig_graph = original_graph                  
        self.mapping    = mapping
        self.adj_graph  = adj_graph
                
        #specific to cora dataset
        self.maximum_dimension = 4
        
        concatenated_features = []
        training_mask = []
        labels = []
        
        for simplex_node_id in adj_graph.nodes():
            simplex_node = self.mapping[int(simplex_node_id)]
            length = len(simplex_node)
            concatenated_feature = torch.zeros(0)
            
            mask_check = True
            for node in simplex_node:
                if node in training_nodes:
                    mask_check = False
                concatenated_feature = torch.cat((concatenated_feature,self.existing_node_features[node],), dim=0)
            if not data_leak:
                if mask_check:
                    training_mask = training_mask + [True]
                else:
                    training_mask = training_mask + [False]
            labels = labels + [original_graph.ndata['label'][simplex_node[-1]]]
                
            remainder = self.maximum_dimension - length + 1
            
            for _ in range(remainder):
                concatenated_feature = torch.cat((concatenated_feature,torch.zeros(self.dimension_of_feature_vectors)), dim=0)
                
            concatenated_features.append(concatenated_feature)
 
        concatenated_features_non_deg = torch.stack(concatenated_features, dim=0)
        self.adj_graph.ndata['feat']      = concatenated_features_non_deg
        if not data_leak:
            self.adj_graph.ndata['train_mask'] = torch.cat((original_graph.ndata['train_mask'], torch.ones(26696, dtype=torch.bool)), dim=0)
        else:
            self.adj_graph.ndata['train_mask'] = torch.tensor(training_mask, dtype=torch.bool)
        #29404 nodes of adj_graph, and 2708 nodes of orig graph, and difference = 26696
        self.adj_graph.ndata['val_mask'] = torch.cat((original_graph.ndata['val_mask'], torch.zeros(26696, dtype=torch.bool)), dim=0)
        self.adj_graph.ndata['test_mask'] = torch.cat((original_graph.ndata['test_mask'], torch.zeros(26696, dtype=torch.bool)), dim=0)
        self.adj_graph.ndata['label'] = torch.tensor(labels)
        save_graphs(tosave, self.adj_graph)