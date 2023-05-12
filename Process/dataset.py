import os
import random

import numpy as np
import torch
from Process.getTwittergraph import Node_tweet
from torch.utils.data import Dataset
from torch_geometric.data import Data


def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Twitter16graph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            tdnew_edgeindex = [row, col]
        else:
            tdnew_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        
        bu_edge_index_ori = [burow,bucol]


        tree = self.treeDic[id]
        index2node = {}
        for i in tree:
            node = Node_tweet(idx=i)
            index2node[i] = node

        for j in tree:
            indexC = j
            indexP = tree[j]['parent']
            nodeC = index2node[indexC]
            if not indexP == 'None':
                nodeP = index2node[int(indexP)]
                nodeC.parent = nodeP
                nodeP.children.append(nodeC)
            else:
                rootindex = indexC - 1
                root_index = nodeC.index
                root_word = nodeC.word

        mask = [0 for _ in range(len(index2node))]
        mask[rootindex] = 1
        root_node = index2node[int(rootindex + 1)]
        que = root_node.children.copy()
        while len(que) > 0:
            cur = que.pop()
            if random.random() >= 0.6:
                mask[int(cur.idx) - 1] = 1
                for child in cur.children:
                    que.append(child)
        if self.tddroprate > 0 and self.budroprate > 0:
            return Data(x=torch.tensor(data['x'],dtype=torch.float32), 
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask = torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                        BU_edge_index_ori=torch.LongTensor(bu_edge_index_ori),TD_edge_index=torch.LongTensor(tdnew_edgeindex),
                 y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']), 
                 rootindex=torch.LongTensor([int(data['rootindex'])]))
        else:
            return Data(x=torch.tensor(data['x'], dtype=torch.float32),
                        x_pos=torch.tensor(data['x_pos'], dtype=torch.float32),
                        mask=torch.tensor(mask, dtype=torch.bool),
                        edge_index=torch.LongTensor(edgeindex),BU_edge_index_ori=torch.LongTensor(bu_edge_index_ori),
                        y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                        rootindex=torch.LongTensor([int(data['rootindex'])]))
