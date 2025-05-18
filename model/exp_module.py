import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from typing import Union, Tuple, Optional,List,Any

from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch.nn import Parameter
# from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

class STAGATE(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(STAGATE, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)
    
class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        #     self.lin_src = Linear(in_channels, heads * out_channels,
        #                           bias=False, weight_initializer='glorot')
        #     self.lin_dst = self.lin_src
        # else:
        #     self.lin_src = Linear(in_channels[0], heads * out_channels, False,
        #                           weight_initializer='glorot')
        #     self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
        #                           weight_initializer='glorot')

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src


        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        # if bias and concat:
        #     self.bias = Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        self._alpha = None
        self.attentions = None

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin_src.reset_parameters()
    #     self.lin_dst.reset_parameters()
    #     glorot(self.att_src)
    #     glorot(self.att_dst)
    #     # zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention


        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # if self.bias is not None:
        #     out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    

import ot
import pandas as pd
import numpy as np
from torch_geometric.nn import VGAE
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import (
    v_measure_score, homogeneity_score, completeness_score)
from sklearn.decomposition import PCA
import scanpy as sc
import torch.backends.cudnn as cudnn
from torch_geometric.nn import SGConv
cudnn.deterministic = True  
cudnn.benchmark = True


class stAA(nn.Module):
    def __init__(self, input_dim, n_clusters, hidden_dim=256, embed_dim=32, 
                 reg_hidden_dim_1=64, reg_hidden_dim_2=32,
                 clamp=0.01, epochs=1000) -> None:
        super(stAA, self).__init__()
        encoder = VariationalEncoder(in_channels=input_dim,
            hidden_channels=hidden_dim, out_channels=embed_dim)
        self.regularizer = Regularizer(embed_dim,
                                       reg_hidden_dim_2, reg_hidden_dim_1)
        self.graph = VGAE(encoder)
        self.ss_classifier = nn.Sequential(nn.Linear(embed_dim,
                                                     n_clusters, bias=False),
                                           nn.Sigmoid())
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.clamp = clamp

    def train_model(self, data,  ss_labels=None, position=None, data_save_path=None, labels=None, print_freq=200, 
                  lr=1e-5, W_a=0.4, W_x=0.6, refine=True, reso=0.5,
                  weight_decay=5e-05, reg_lr=1e-5, eval=True):
        loss_func = nn.CrossEntropyLoss()
        encoder_optimizer = torch.optim.Adam([{'params': self.graph.encoder.parameters()},
                                              {"params": self.ss_classifier.parameters()}
                                              ],
                                             lr=lr, weight_decay=weight_decay)
        regularizer_optimizer = torch.optim.Adam(self.regularizer.parameters(),
                                                 lr=reg_lr)
        data = data.cuda()
        if np.min(ss_labels) == 1:
            ss_labels = ss_labels - 1
        ss_labels = torch.tensor(ss_labels, dtype=torch.int64).cuda()
        for epoch in range(self.epochs):
            self.train()
            encoder_optimizer.zero_grad()

            z = self.graph.encode(data.x, data.train_pos_edge_index)

            for i in range(1):
                f_z = self.regularizer(z)
                r = torch.normal(
                    0.0, 1.0, [data.num_nodes, self.embed_dim]).cuda()
                f_r = self.regularizer(r)
                reg_loss = - f_r.mean() + f_z.mean()
                regularizer_optimizer.zero_grad()
                reg_loss.backward(retain_graph=True)
                regularizer_optimizer.step()

                for p in self.regularizer.parameters():
                    p.data.clamp_(-self.clamp, self.clamp)
    
            f_z = self.regularizer(z)
            generator_loss = -f_z.mean()
            adj_recon_loss = self.graph.recon_loss(
                z, data.train_pos_edge_index) + (1 / data.num_nodes) * self.graph.kl_loss()
            adj_recon_loss = (adj_recon_loss + generator_loss) * W_a

            output_ss = self.ss_classifier(z)
            X_recon_loss = loss_func(output_ss, ss_labels) * W_x

            loss = X_recon_loss+adj_recon_loss
            loss.backward()
            encoder_optimizer.step()
            if (epoch+1) % print_freq == 0:
                print('Epoch: {:03d}, Loss: {:.4f}, ADJ Loss: {:.4f}, Gene Loss: {:.4f}'.format(
                    epoch+1, float(loss.item()), float(adj_recon_loss.item()), float(X_recon_loss.item())))

        completeness, hm, nmi, ari, z, pca_embedding, pred_label = self.eval_model(
                data, labels=labels, refine=refine, position=position,
                save_name=data_save_path, reso=reso)
        res = {}
        res["embedding"] = z
        res["pred_label"] = pred_label
        res["embedding_pca"] = pca_embedding
        if eval == True:
            res["nmi"] = nmi
            res["ari"] = ari
            res["completeness"] = completeness
            res["hm"] = hm

        return res


    @torch.no_grad()
    def eval_model(self, data, labels=None, refine=False, reso=0.5,
                   position=None, save_name=None):
        self.eval()
        z = self.graph.encode(data.x, data.train_pos_edge_index)
        pca_input = dopca(z.cpu().numpy(), dim=20)
        adata_tmp=sc.AnnData(pca_input)
        sc.pp.neighbors(adata_tmp, n_neighbors=20)
        sc.tl.louvain(adata_tmp, resolution=reso, random_state=0)
        pred_mclust=adata_tmp.obs['louvain'].astype(int).to_numpy()
        if refine:
            pred_mclust = refine_label(pred_mclust, position, radius=50)

        if labels is not None:
            label_df = pd.DataFrame({"True": labels,
                                    "Pred": pred_mclust}).dropna()
            # label_df = pd.DataFrame({"True": labels, "Pred": pred}).dropna()
            completeness = completeness_score(
                label_df["True"], label_df["Pred"])
            hm = homogeneity_score(label_df["True"], label_df["Pred"])
            nmi = v_measure_score(label_df["True"], label_df["Pred"])
            ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
        else:
            completeness, hm, nmi, ari = 0, 0, 0, 0

        if (save_name is not None):
            np.save(save_name+"pca.npy", pca_input)
            np.save(save_name+"embedding.npy", z.cpu().numpy())
            if (labels is not None):
                pd.DataFrame({"True": labels, 
                            "Pred": pred_mclust}).to_csv(save_name+'types.txt')
            else:
                pd.DataFrame({
                            "Pred": pred_mclust}).to_csv(save_name+'types.txt')

        return completeness, hm, nmi, ari, z.cpu().numpy(), pca_input, pred_mclust

class Regularizer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Regularizer, self).__init__()
        self.den1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.den1.bias.data.fill_(0.0)
        self.den1.weight.data = torch.normal(0.0, 0.001, [hidden_dim1, input_dim])
        self.den2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.den2.bias.data.fill_(0.0)
        self.den2.weight.data = torch.normal(0.0, 0.001, [hidden_dim2, hidden_dim1])
        self.output = torch.nn.Linear(hidden_dim2, 1)
        self.output.bias.data.fill_(0.0)
        self.output.weight.data = torch.normal(0.0, 0.001, [1,hidden_dim2])
        self.act = torch.sigmoid
    def forward(self, inputs):
        dc_den1 = self.act(self.den1(inputs))
        dc_den2 = torch.sigmoid((self.den2(dc_den1)))
        output = self.output(dc_den2)
        return output


class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(VariationalEncoder,self).__init__()
        self.conv1 = SGConv(in_channels, hidden_channels)
        self.conv_mu = SGConv(hidden_channels, out_channels)
        self.conv_logstd = SGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)  

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=42)
    X_10 = pcaten.fit_transform(X)
    return X_10

def refine_label(label, position, 
                 radius=50):
    new_type = []

    # calculate distance
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, radius+1):
            neigh_type.append(label[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type