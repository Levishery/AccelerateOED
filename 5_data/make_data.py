import torch
import numpy as np
import demjson
import random
from torch_geometric.data import Data



def getEdgeAtt(attr1, attr2, n):
    edge_attr = torch.zeros([2, n * (n - 1)])
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr

# fileObject = open('data_5_oscillators_45000.json', 'r')
# data_ = fileObject.read()
# data_ = demjson.decode(data_)
# fileObject = open('data_5_oscillators3.json', 'r')
# data_1 = fileObject.read()
# data_1 = demjson.decode(data_1)
# fileObject = open('data_5_oscillators4.json', 'r')
# data_2 = fileObject.read()
# data_2 = demjson.decode(data_2)
# data = data_+data_1+data_2

fileObject = open('data_5_oscillators3.json', 'r')
data2 = fileObject.read()
data2 = demjson.decode(data2)

fileObject = open('data/data_5_oscillators.json', 'r')
data1 = fileObject.read()
data1 = demjson.decode(data1)

data = data1+data2
random.shuffle(data)

data_list = []
# fully connected


for i in range(len(data)):

    x = np.asarray(data[i]['w'])
    x = torch.from_numpy(x.astype(np.float32))
    n = x.size()[0]
    x = x.unsqueeze(dim=1)

    edge_index = getEdgeAtt(np.tile(np.asarray([i for i in range(n)]), (n, 1)),
                            np.tile(np.asarray([[i] for i in range(n)]), (1, n)), n).long()
    edge_attr = getEdgeAtt(torch.from_numpy(np.asarray(data[i]['a_lower']).astype(np.float32)),
                           torch.from_numpy(np.asarray(data[i]['a_upper']).astype(np.float32)), n)

    y = torch.from_numpy(np.asarray(data[i]['mean_MOCU']).astype(np.float32))
    y = y.unsqueeze(dim=0).unsqueeze(dim=0)

    data_ = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
    data_list.append(data_)

torch.save(data_list, 'TensorDataSet_95000_mixed.pth'.replace('Tensor', 'Graph'))
