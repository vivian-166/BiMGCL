import sys, os

sys.path.append(os.getcwd())
from Process.process import *
import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_mean

from Process.rand5fold import *
from tools.evaluate import *
from tools.aug import *
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
import copy
import math

import random
from torch_geometric.utils import subgraph


class PriorDiscriminator(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = th.nn.Linear(input_dim, input_dim)
        self.l1 = th.nn.Linear(input_dim, input_dim)
        self.l2 = th.nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return th.sigmoid(self.l2(h))


class FF(th.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = th.nn.Sequential(
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU(),
            th.nn.Linear(input_dim, input_dim),
            th.nn.ReLU()
        )
        self.linear_shortcut = th.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class Encoder(th.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.num_features = num_features
        self.dim = dim    
        self.convs = th.nn.ModuleList()
        self.bns = th.nn.ModuleList()

        for i in range(num_gc_layers):
            if i > 1:
                nn = Sequential(Linear(dim*i, dim), ReLU(), Linear(dim, dim))
            elif i:
                nn = Sequential(Linear(dim+self.num_features, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = th.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, rootindex):

        x_one = copy.deepcopy(x)
        xs_one = []
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        x, edge_index = x.to(device), edge_index.to(device)
        x1 = copy.copy(x.float())
        x = self.convs[0](x, edge_index)
        x2 = copy.copy(x)
        x = F.relu(x)
        xs_one.append(x)
        rootindex = rootindex
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        root_extend = th.zeros(len(batch), x1.size(1)).to(device)
        batch_size = max(batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.convs[1](x, edge_index)
        x3 = copy.copy(x)
        x = F.relu(x)
        xs_one.append(x)
        root_extend = th.zeros(len(batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.convs[2](x, edge_index)
        x = F.relu(x)
        xs_one.append(x)
        root_extend = th.zeros(len(batch), x3.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(batch, num_batch))
            root_extend[index] = x3[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, batch, dim=0)

        xpool_one = [global_mean_pool(x_one, batch) for x_one in xs_one] 
        x_one = th.cat(xpool_one, 1)
        return x_one, th.cat(xs_one, 1)

    def get_embeddings(self, data, str=None):

        with th.no_grad():
            if str=='bu':
                x, edge_index, batch, rootindex = data.x, data.BU_edge_index_ori, data.batch, data.rootindex
                graph_embed, node_embed = self.forward(x, edge_index, batch, rootindex)
            else:
                x, edge_index, batch, rootindex = data.x, data.edge_index, data.batch, data.rootindex
                graph_embed, node_embed = self.forward(x, edge_index, batch, rootindex)

        return node_embed

def sim(h1, h2): 
    z1 = F.normalize(h1, dim=-1, p=2)
    z2 = F.normalize(h2, dim=-1, p=2)
    # return th.mm(z1, z2.t())
    if len(z1.shape) >= 2:
        return th.mm(z1, z2.t())
    else:
        return (z1*z2).sum()

def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)
    Ep = log_2 - F.softplus(- p_samples)
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    Eq = F.softplus(-q_samples) + q_samples

    if average:
        return Eq.mean()
    else:
        return Eq

def graph_level_loss_(td_ori, td_aug, bu_ori, bu_aug):
    
    td_sim = sim(td_ori, td_aug)
    bu_sim = sim(bu_ori, bu_aug)
    cross_sim = sim(th.cat([td_ori, bu_ori], -1), th.cat([td_aug, bu_aug], -1))
    same_graph = (torch.diag(td_sim).sum() + torch.diag(bu_sim).sum() + torch.diag(cross_sim).sum()) /len(td_ori) # 同一个图的表示拉近
    diff_graph = (td_sim.sum() + bu_sim.sum() + cross_sim.sum() - same_graph) / 2 / len(td_ori) / len(td_ori)
    
    return F.softplus(-same_graph) + F.softplus(diff_graph)

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_dropped_two, l_enc_pos=None, l_enc_dropped=None):
    '''
    Args:
        l: Local feature map.
        g: Global features.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    pos_mask = th.zeros((num_nodes, num_graphs), device=device)
    neg_mask = th.ones((num_nodes, num_graphs), device=device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = th.mm(l_enc, g_enc.t())
    pos = res
    if l_enc_pos:
        res_two = th.mm(l_enc_pos, g_enc.t())
        pos += res_two
    if l_enc_dropped:
        res_three = th.mm(l_enc_dropped, g_enc.t())
        pos += res_three
    res_four = th.mm(l_enc_dropped_two, g_enc.t())
    pos += res_four

    E_pos = get_positive_expectation(pos * pos_mask / 4, measure,
                                     average=False).sum()
    E_pos = E_pos / num_nodes

    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


class Net(th.nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(Net, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.td_encoder = Encoder(5000, hidden_dim, num_gc_layers)
        self.bu_encoder = Encoder(5000, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim*2)
        self.global_d = FF(self.embedding_dim*2)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, th.nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, rootindex):

        x,        edge_index,       buedge_index_ori,      bu_edge_index,     td_edge_index,        batch,       num_graphs,       mask = \
        data.x, data.edge_index, data.BU_edge_index_ori, data.BU_edge_index, data.TD_edge_index, data.batch, max(data.batch) + 1, data.mask
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        td_edge_sub_ori, _ = subgraph(mask, edge_index)
        node_mask = th.ones((x.size(0), 1), device=device)
        for i in range(x.size(0)):
            if random.random() >= 0.8:
                node_mask[i] = 0
        node_mask[data.rootindex] = 1
        x_pos_two = x * node_mask 
        bu_edge_index, td_edge_index = aug_random_edge(bu_edge_index, td_edge_index, len(x))

        # td
        td_y, td_M = self.td_encoder(x, edge_index, batch, rootindex)
        td_y_dropped_two, td_M_dropped_two = self.td_encoder(x_pos_two, td_edge_index, batch, rootindex)

        # bu
        bu_edge_sub_ori, _ = subgraph(mask.t(), buedge_index_ori) 
        bu_y, bu_M = self.bu_encoder(x, buedge_index_ori, batch, rootindex)
        bu_y_dropped_two, bu_M_dropped_two = self.bu_encoder(x_pos_two, bu_edge_index, batch, rootindex)

        g_enc = self.global_d(th.cat((td_y, bu_y), dim=1))
        l_enc = self.local_d(th.cat((td_M, bu_M), dim=1)) 

        l_enc_dropped_two = self.local_d(th.cat((td_M_dropped_two, bu_M_dropped_two), dim=1))

        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, l_enc_dropped_two)

        graph_level_loss = graph_level_loss_(td_y, td_y_dropped_two, bu_y, bu_y_dropped_two)
        return local_global_loss + graph_level_loss
        
    def get_embedding(self, Batch_data):
        td_emb = self.td_encoder.get_embeddings(Batch_data, 'td')
        bu_emb = self.bu_encoder.get_embeddings(Batch_data, 'bu')
        return th.cat((td_emb,bu_emb), dim=1)


class Classfier(th.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes, td_encoder, bu_encoder):
        super(Classfier, self).__init__()
        self.linear_one = th.nn.Linear(5000 * 2 + 0 * hid_feats, 4 * hid_feats)
        self.linear_two = th.nn.Linear(hid_feats * 4, hid_feats)
        self.linear_three = th.nn.Linear(in_feats, hid_feats)
        self.linear_four = th.nn.Linear(6 * hid_feats, hid_feats * 4)

        self.linear_transform = th.nn.Linear(hid_feats * 2, 4)
        self.prelu = th.nn.PReLU()
        self.td_encoder = td_encoder
        self.bu_encoder = bu_encoder

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_x, data_batch, data_rootindex, edge_index):
        ori = scatter_mean(data_x, data_batch, dim=0)
        root = data_x[data_rootindex]
        ori = th.cat((ori, root), dim=1)
        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        _, td_emb = self.td_encoder(data_x, edge_index, data_batch, data_rootindex)
        _, bu_emb = self.bu_encoder(data_x, edge_index, data_batch, data_rootindex)
        embed = th.cat((td_emb, bu_emb), dim=1)
        x = scatter_mean(embed, data_batch, dim=0)

        x = self.linear_four(x)
        x =  F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        x = self.linear_three(x)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = th.cat((x, ori), dim=1)
        out = self.linear_transform(out)
        x = F.log_softmax(out, dim=1)
        return x
