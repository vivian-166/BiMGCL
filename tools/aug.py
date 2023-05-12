import torch
import copy
import random
import scipy.sparse as sp
import numpy as np


def aug_random_mask(input_feature, drop_percent=0.2):
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(bu_edge_index, td_edge_index, x_len, drop_percent = 0.2):
    percent = drop_percent / 2
    bu_row_idx, bu_col_idx = bu_edge_index
    td_row_idx, td_col_idx = td_edge_index
    length = len(bu_row_idx)

    poslist = random.sample(range(length), int(length * (1 - percent)))
    poslist = sorted(poslist)
    bu_row = list(bu_row_idx[poslist])
    bu_col = list(bu_col_idx[poslist])
    td_row = list(td_row_idx[poslist])
    td_col = list(td_col_idx[poslist])

    bu_row2col = {(f,s):1 for f, s in zip(bu_row, bu_col)}
    td_row2col = {(f,s):1 for f, s in zip(td_row, td_col)}
    for _ in range(int(length * percent)):
        add_row = random.randint(0, x_len-1)
        add_col = random.randint(0, x_len-1)
        while(bu_row2col.get((add_row, add_col)) == 1):
            add_row = random.randint(0, x_len-1)
            add_col = random.randint(0, x_len-1)
        bu_row2col[(add_row, add_col)] = 1
        td_row2col[(add_col, add_row)] = 1

    bu_row, bu_col, td_row, td_col = [], [], [], []
    for k, _ in bu_row2col.items():
        bu_row.append(k[0])
        bu_col.append(k[1])
    for k, _ in td_row2col.items():
        td_row.append(k[0])
        td_col.append(k[1])
        
    return torch.LongTensor([bu_row, bu_col]), torch.LongTensor([td_row, td_col])


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break
  
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_feature_dropout(input_feat, drop_percent = 0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


def aug_feature_dropout_cell(input_feat, drop_percent = 0.2):
    aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    input_feat_dim = aug_input_feat.shape[1]
    num_of_nodes = aug_input_feat.shape[0]   
    drop_feat_num = int(num_of_nodes * input_feat_dim * drop_percent)
    
    position = []
    number_list = [j for j in range(input_feat_dim)]
    for i in range(num_of_nodes):
      number_i = [i for k in range(input_feat_dim)]
      position += list(zip(number_i, number_list))
      
    drop_idx = random.sample(position, drop_feat_num)
    for i in range(len(drop_idx)):
        aug_input_feat[(drop_idx[i][0],drop_idx[i][1])] = 0.0
    
    return aug_input_feat


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S


def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out