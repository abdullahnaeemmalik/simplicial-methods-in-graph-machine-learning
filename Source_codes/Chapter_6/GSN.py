"""Modified from Torch Module for Topology Adaptive Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import dgl
from dgl import function as fn

class GSN(nn.Module):
    r"""
    .. math::
        H^{K} = {\sum}_{k=0}^K (D^{-1/2} A D^{-1/2})^{k} X W_{k},

    where :math:`A` denotes the adjacency matrix,
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix,
    :math:`W_{k}` denotes the linear weights to sum the results of different hops together.

    Parameters
    ----------
    in_feats : int
        Input feature size. i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Output feature size.  i.e, the number of dimensions of :math:`H^{K}`.
    maximum_dim : int, required
        Equivalent to number of hops.
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
        super(GSN, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._maximum_dim = maximum_dim
        self._activation = activation
        self.weight_bool = weight
        if self.weight_bool:
            self.lin = nn.Linear(in_feats * 2 * (self._maximum_dim + 1), out_feats, bias=bias)

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

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Gated Simplicial Convolution

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
        with graph.local_scope():
            norm = th.pow(graph.in_degrees().to(feat).clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            graph = dgl.reverse(graph, copy_edata=True, copy_ndata=True)
            msg_func = fn.copy_u("h", "m")

            fstack = [feat]
            for _ in range(self._maximum_dim):
                rst = fstack[-1] * norm
                graph.ndata["h"] = rst
                graph.update_all(msg_func, fn.sum(msg="m", out="h"))
                rst = graph.ndata["h"]
                rst = rst * norm
                fstack.append(rst)
            
            graph = dgl.reverse(graph, copy_edata=True, copy_ndata=True)
                
            for _ in range(self._maximum_dim):
                rst = fstack[-1] * norm
                graph.ndata["h"] = rst
                graph.update_all(msg_func, fn.sum(msg="m", out="h"))
                rst = graph.ndata["h"]
                rst = rst * norm
                fstack.append(rst)

            if self.weight_bool:
                rst = self.lin(th.cat(fstack, dim=-1))

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
