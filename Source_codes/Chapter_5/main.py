import dgl
import torch
import os

from torch.sparse import *
import collections
os.environ['DGLBACKEND'] = 'pytorch'
#from dgl.nn import GraphConv, SAGEConv
from pathlib import Path
import GCN_modified
import torch.nn as nn
import torch.nn.functional as F
from utils import plot_losses, train, train_step, get_experiment_stats
from utils import test, characterize_performance, norm_plot
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import json

def load_dicts(file_path):
    with open(file_path, "r") as file:
        loaded_data = json.load(file)
        
    partitioned_tv = {int(key): torch.tensor(value)
                      for key, value in loaded_data["partitioned_tv"].items()}
        
    partition_indices_and_one_hot_tv = {int(key): torch.tensor(value)
                                        for key, value in
                                        loaded_data["partition_indices_and_one_hot_tv"].items()}
        
    multi_hot_tv_dict = {int(key): torch.tensor(value)
                         for key, value in loaded_data["multi_hot_tv_dict"].items()}
        
    partition_times_hot_dict = {int(key): torch.tensor(value)
                                for key, value in loaded_data["partition_times_hot_dict"].items()}
    
    return multi_hot_tv_dict, partition_indices_and_one_hot_tv, partitioned_tv
    


class GCN(nn.Module):
    def __init__(self, graph, input_layer:int, hidden_layers:int, output_layer:int, num_layers:int, dropout, alpha, beta):
        self.input_layer   = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer  = output_layer
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.graph         = graph
        super(GCN, self).__init__()
        self.convs         = nn.ModuleList()
        self.bns           = torch.nn.ModuleList()
        self.alpha         = alpha
        self.beta          = beta
        self.bns.append(torch.nn.BatchNorm1d(hidden_layers))
        self.convs.append(GCN_modified.GraphConv(input_layer, hidden_layers, alpha=self.alpha, beta=self.beta, weight=True, bias=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCN_modified.GraphConv(hidden_layers, hidden_layers,alpha = self.alpha, beta=self.beta, weight=True, bias=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_layers))
        self.convs.append(GCN_modified.GraphConv(hidden_layers, output_layer, alpha=self.alpha, beta=self.beta, weight=True, bias=True))

    def forward(self, graph, input_features):
        for conv in self.convs[:-1]:
            input_features = conv(graph, input_features)
            input_features = F.relu(input_features)
            input_features = F.dropout(input_features, p=self.dropout, training=self.training)
        input_features = self.convs[-1](graph, input_features)
        return input_features.log_softmax(dim=-1)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
arxiv_graph_raw=dataset.graph[0]
split_idx = dataset.get_idx_split()
labels = dataset.labels.flatten().to(device)
val_acc_lb, val_acc_lb_var, test_acc_lb, test_acc_lb_var = 0.7300, 0.0017, 0.7174, 0.0029
evaluator = Evaluator(name = "ogbn-arxiv")

    
file_path = "arxiv_semi_tv_upto5.json"
print("Loading vertex features for the graph now from json file")

multi_hot_tv_dict, partition_indices_and_one_hot_tv, partitioned_tv = load_dicts(file_path)


def add_features_to_graph(graph,features):
    """All the existing features and new features need to be stacked and then replaced
    with existing features. DGL graph library doesn't support adding node features node-wise
    Returns the graph"""
    if not "feat" in graph.ndata:
        print("Error! The graph should have node features called 'feat'.")
        return
    else:
        existing_node_features = graph.ndata['feat']
    
    if not (type(features) == dict or type(features) == collections.defaultdict):
        print("Error! Features should be a dictionary")
        return
    stacked_features = torch.stack(tuple(features.values()))
    # Iterate through nodes and concatenate stacked_features to existing_node_features
    concatenated_features = []
    for node_id in graph.nodes():
        concatenated_feature = torch.cat((existing_node_features[node_id], stacked_features[node_id]), dim=0)
        concatenated_features.append(concatenated_feature)

    concatenated_features = torch.stack(concatenated_features, dim=0)

    graph.ndata['feat'] = concatenated_features
    
    return graph

"""multi_hot_tv_dict dictionary Type I

partition_indices_and_one_hot_tv Type II

partitioned_tv dictionary Type III
"""

print("Augmenting features of the graph..")
arxiv_graph_bidirected = dgl.to_bidirected(arxiv_graph_raw,copy_ndata =True)
arxiv_graph_bidirected = dgl.add_self_loop(arxiv_graph_bidirected)
#arxiv_graph_bidirected_type0 = arxiv_graph_bidirected
"""Replace name of graph and features to add"""
arxiv_graph_bidirected_typeIII = add_features_to_graph(
    arxiv_graph_bidirected,partitioned_tv)

def getdim_inlayer(graph):
    """The dimensions of each node features is different, so 
    a function here is needed that can yield the dimension of the input layer"""
    return graph.ndata['feat'].size()[1]

"""Replace name of graph and alpha beta values"""
model_kwargs = dict(graph=arxiv_graph_bidirected_typeIII, input_layer=getdim_inlayer(arxiv_graph_bidirected_typeIII), 
                     hidden_layers=256, output_layer=40, 
                 num_layers=3, dropout=0.5, alpha = 1, beta = 0)
model = GCN(**model_kwargs).to(device)

# Where to save the best model
model_path = 'models'
Path(model_path).mkdir(parents=True, exist_ok=True)
gcn_path = f"{model_path}/gcn_typeIII_bidirected_a1_b0.model"

"""Replace name of graph"""
train_args = dict(
    graph=arxiv_graph_bidirected_typeIII, labels=labels, split_idx=split_idx, 
    epochs=500, evaluator=evaluator, device=device, 
    save_path=gcn_path, lr=5e-3, es_criteria=50,
)

"""Replace name of graph and features to add"""
print("Training for arxiv_graph_bidirected_typeIII with I + A +A^T norm = left using beta = 0 and alpha = 1")
train_losses, val_losses = train(model=model, verbose=True, **train_args)
plot_losses(train_losses, val_losses, log=True, modelname='typeIII_I+A+AT_a1_b0')

"""Replace name of graph"""
_ = characterize_performance(model, arxiv_graph_bidirected_typeIII, labels, split_idx, evaluator, gcn_path, verbose=True)