import torch
import numpy as np
import demjson
import random
from torch_geometric.data import Data


fileObject = open('data/data_7_oscillators.json', 'r')
data2 = fileObject.read()
data2 = demjson.decode(data2)

fileObject = open('data/data_7_oscillators_1.json', 'r')
data1 = fileObject.read()
data1 = demjson.decode(data1)

fileObject = open('data/data_7_oscillators_2.json', 'r')
data3 = fileObject.read()
data3 = demjson.decode(data3)

fileObject = open('data_7_oscillators_5.json', 'r')
data4 = fileObject.read()
data4 = demjson.decode(data4)

data = data1 + data2 + data3 + data4

random.shuffle(data)

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

torch.save(data_list, 'TensorDataSet_20000_7.pth'.replace('Tensor', 'Graph'))