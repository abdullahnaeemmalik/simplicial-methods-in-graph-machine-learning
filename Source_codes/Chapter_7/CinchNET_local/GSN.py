"""Modified from Torch Module for Topology Adaptive Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
from dgl import function as fn
from ConnectingEnvelopes import SimplicialFinder
from AdjFunctor import AdjFunctor

class CinchNETConv(nn.Module):
    r"""
    .. math::
        H^{K} = {\sum}{j=0}^K (D^{-1/2} A D^{-1/2})^{j} W_{j} ({\sum}_{k=0}^K (D^{-1/2} A^T D^{-1/2})^{k} X W_{k}),

    where :math:`A` denotes the adjacency matrix consisting of only those edges that correspond to face maps, and A^T is the transpose of the matrix
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix,
    :math:`W_{k}` denotes the linear weights to sum the results of different hops together.

    Parameters
    ----------
    in_feats : int
        Input feature size. i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Output feature size.  i.e, the number of dimensions of :math:`H^{K}`.
    maximum_dim : int, required
        Sets highest dimension of simplex to be looked for.
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    weigh: Bool, optional
        If True, add a learnable weight matrix via the learnable linear module lin     

    Attributes
    ----------
    lin : torch.Module
        The learnable linear module.

    """

    def __init__(
        self,
        in_feats,
        out_feats,
        maximum_dim,
        bias=True,
        activation=None,
        weight=True,
    ):
        super(CinchNETConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._maximum_dim = maximum_dim
        self._activation = activation
        self.weight_bool = weight
        if self.weight_bool:
            self.lin = nn.Linear(in_feats * 2 * (self._maximum_dim + 1), out_feats, bias=bias)
        self.adj_graph = dgl.heterograph({('non-degenerate', 'd', 'non-degenerate'): ([], [])})

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        
    def create_adj_graph(self,graph):
        """
        Description
        -----------
        Creates adjancency graph of a given a subgraph block
        
        Parameters
        ----------
        graph : DGLGraph (is_block == True)
            The for which its adjancecy graph needs to be computed.
        
        Output
        ------
        Adjacency graph and a list of original nodes of the graph
        
        """
        heavy_duty=SimplicialFinder(graph)
        
        for index in range(self._maximum_dim-1):
            heavy_duty.connectivity_update()
            
        simplex_dict = heavy_duty.standard_0_skeleton_dict
        adj_graph_object = AdjFunctor(simplex_dict,deg_simplices=False)
        adj_graph_object.fill_edges()
        self._maximum_dim = min(adj_graph_object.max_dim,self._maximum_dim)
        src, dst = graph.edges()
        src = src.tolist()
        dst = dst.tolist()
        set_of_nodes = set(src+dst)
        seed_nodes = list(set_of_nodes)
        node_simplex_dict = adj_graph_object.adj_graph_node_dict
        
        #add features to adj_graph
        
        concatenated_features = []
        
        for simplex_node_id in adj_graph_object.adj_graph.nodes():
            simplex_node = node_simplex_dict[int(simplex_node_id)]
            length = len(simplex_node)
            concatenated_feature = th.zeros(0)
            
            for node in simplex_node:
                concatenated_feature = th.cat((concatenated_feature,graph.ndata['feat'][node],), dim=0)
                
            remainder = self._maximum_dim - length + 1
            
            for _ in range(remainder):
                concatenated_feature = th.cat((concatenated_feature,th.zeros(self._in_feats)), dim=0)
                
            concatenated_features.append(concatenated_feature)
            
        concatenated_features_non_deg = th.stack(concatenated_features, dim=0)
        adj_graph_object.adj_graph.ndata['feat'] = concatenated_features_non_deg
        
        return adj_graph_object.adj_graph, seed_nodes
        


    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute CinchNET convolution, formerly known as Gated Simplicial Convolution

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        
        adj_graph, seed_nodes = self.create_adj_graph(graph)
        with adj_graph.local_scope():
            adj_graph = dgl.add_self_loop(adj_graph)
            norm = th.pow(adj_graph.in_degrees().to(feat).clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            adj_graph = dgl.reverse(adj_graph, copy_edata=True, copy_ndata=True)
            msg_func = fn.copy_u("h", "m")

            fstack = [feat]
            for _ in range(self._maximum_dim):
                rst = fstack[-1] * norm
                adj_graph.ndata["h"] = rst
                adj_graph.update_all(msg_func, fn.sum(msg="m", out="h"))
                rst = adj_graph.ndata["h"]
                rst = rst * norm
                fstack.append(rst)
            
            adj_graph = dgl.reverse(adj_graph, copy_edata=True, copy_ndata=True)
                
            for _ in range(self._maximum_dim):
                rst = fstack[-1] * norm
                adj_graph.ndata["h"] = rst
                adj_graph.update_all(msg_func, fn.sum(msg="m", out="h"))
                rst = adj_graph.ndata["h"]
                rst = rst * norm
                fstack.append(rst)

            if self.weight_bool:
                rst = self.lin(th.cat(fstack, dim=-1))

            if self._activation is not None:
                rst = self._activation(rst)
                
            seednodes_tensor = th.tensor(seed_nodes)
            return rst[seednodes_tensor]