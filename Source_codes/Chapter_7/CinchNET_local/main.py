import dgl
import torch
import os

from torch.sparse import *
os.environ['DGLBACKEND'] = 'pytorch'
#from dgl.nn import GraphConv, SAGEConv
from pathlib import Path
import GSN
import torch.nn as nn
import torch.nn.functional as F
from utils import plot_losses, train, train_step, get_experiment_stats
from utils import test, characterize_performance, norm_plot
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from scipy import stats


file_path_for_adj_graph_with_features = "./arxiv_adj_graph_with_features"
arxiv_adj_graph, label_dict = dgl.load_graphs(file_path_for_adj_graph_with_features)
arxiv_adj_graph = arxiv_adj_graph[0]   


class BnNN(nn.Module):
    def __init__(self, graph, input_layer:int, hidden_layers:int, output_layer:int, num_layers:int, dropout):
        self.input_layer   = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer  = output_layer
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.graph         = graph
        super(BnNN, self).__init__()
        self.convs         = nn.ModuleList()
        self.bns           = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_layers))
        maximum_dim = 2
        self.convs.append(GSN.GSN(input_layer, hidden_layers, maximum_dim, bias=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GSN.GSN(hidden_layers, hidden_layers, maximum_dim, bias=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_layers))
        self.convs.append(GSN.GSN(hidden_layers, output_layer, maximum_dim, bias=True))

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
split_idx = dataset.get_idx_split()
labels = dataset.labels.flatten().to(device)
val_acc_lb, val_acc_lb_var, test_acc_lb, test_acc_lb_var = 0.7300, 0.0017, 0.7174, 0.0029
evaluator = Evaluator(name = "ogbn-arxiv")

def getdim_inlayer(graph):
    """The dimensions of each node features is different, so 
    a function here is needed that can yield the dimension of the input layer"""
    return graph.ndata['feat'].size()[1]

model_kwargs = dict(graph=arxiv_adj_graph, input_layer=getdim_inlayer(arxiv_adj_graph), 
                     hidden_layers=512, output_layer=40, 
                 num_layers=3, dropout=0.5)
model = BnNN(**model_kwargs).to(device)

# Where to save the best model
model_path = 'models'
Path(model_path).mkdir(parents=True, exist_ok=True)
gcn_path = f"{model_path}/gcn_typeI_bidirected_a1_b0.model"

"""Replace name of graph"""
train_args = dict(
    graph=arxiv_adj_graph, labels=labels, split_idx=split_idx, 
    epochs=200, evaluator=evaluator, device=device, 
    save_path=gcn_path, lr=5e-3, es_criteria=50,
)

"""Replace name of graph and features to add"""
print("Training for arxiv_graph_bidirected_type0 with I + A +A^T norm = left using beta = 0 and alpha = 1")
train_losses, val_losses = train(model=model, verbose=True, **train_args)
plot_losses(train_losses, val_losses, log=True, modelname='typeI_I+A+AT_a1_b0')

"""Replace name of graph"""
_ = characterize_performance(model, arxiv_adj_graph, labels, split_idx, evaluator, gcn_path, verbose=True)

print("Beginning training with 10 experiments of the model for directed graph with mixed tv..")
df_gcn = get_experiment_stats(
    model_cls=BnNN, model_args=model_kwargs,
    train_args=train_args, n_experiments=10
)

print("Creating norm plot for directed graph with tv and pi")
norm_plot(
    [
        (test_acc_lb, test_acc_lb_var, 'Leaderboard'), 
        (df_gcn.loc['mean', 'test_acc'], df_gcn.loc['std', 'test_acc'], 'GCN'),
    ],
    'Test Performance'
)

print("Evaluating the model for directed graph with refined tv")
_, p = stats.ttest_ind_from_stats(
    test_acc_lb, test_acc_lb_var, 10,
    df_gcn.loc['mean', 'test_acc'], df_gcn.loc['std', 'test_acc'], 10,
    equal_var=False,
)
print(f"Mean Test Accuracy Improvement: {(df_gcn.loc['mean', 'test_acc'] - test_acc_lb):.4f}")
print(f"Probability that these are from the same performance distribution = {p*100:.0f}%")