"""Torch modules for graph convolutions(GCN) modified to account for parameters alpha and beta."""
import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair


class GraphConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        alpha, beta,
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GraphConv, self).__init__()
        if beta < 0 or beta > 1:
            raise DGLError(
                'Invalid beta value. Must be between 0 and 1'
                ' But got "{}".'.format(beta)
            )
        if alpha < 0 or alpha > 1:
            raise DGLError(
                'Invalid alpha value. Must be between 0 and 1'
                ' But got "{}".'.format(alpha)
            )
        parameter_check = alpha + beta
        if parameter_check != 1:
            raise DGLError(
                'Invalid alpha, beta values. The sum of alpha={} and beta={} must be 1.'
                .format(alpha, beta)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._alpha = alpha
        self._beta = beta
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation
        
    def reset_parameters(self):

        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value
        
    def forward(self, graph, feat, weight=None, edge_weight=None):
        
        with graph.local_scope():
            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            feat_src, feat_dst = expand_as_pair(feat, graph)
            degs_out = graph.out_degrees().to(feat_src).clamp(min=1)
            degs_in = graph.in_degrees().to(feat_dst).clamp(min=1)
            degs_in = th.pow(degs_in, -self._alpha)
            degs_out = th.pow(degs_out, -self._beta)
            degs = degs_in * degs_out
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)
