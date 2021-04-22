import torch
import numpy as np
import demjson
import random
from torch_geometric.data import Data


fileObject = open('../Dataset/5o_type1.json', 'r')
data2 = fileObject.read()
data2 = demjson.decode(data2)

fileObject = open('../Dataset/5o_type2.json', 'r')
data1 = fileObject.read()
data1 = demjson.decode(data1)

# fileObject = open('../Dataset/7o_type1.json', 'r')
# data2 = fileObject.read()
# data2 = demjson.decode(data2)
#
# fileObject = open('../Dataset/7o_type2.json', 'r')
# data1 = fileObject.read()
# data1 = demjson.decode(data1)

data = data1 + data2

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

    # data.x: Node feature matrix with shape[num_nodes, num_node_features]
    #
    # data.edge_index: Graph connectivity in COO format with shape[2, num_edges] and type torch.long
    #
    # data.edge_attr: Edge feature matrix with shape[num_edges, num_edge_features]
    #
    # data.y: Target to train against(may have arbitrary shape), e.g., node - level targets of shape[num_nodes, *]
    # or graph - level targets of shape[1, *]
    # Normalize targets to mean = 0 and std = 1.

    edge_index = getEdgeAtt(np.tile(np.asarray([i for i in range(n)]), (n, 1)),
                            np.tile(np.asarray([[i] for i in range(n)]), (1, n)), n).long()
    edge_attr = getEdgeAtt(torch.from_numpy(np.asarray(data[i]['a_lower']).astype(np.float32)),
                           torch.from_numpy(np.asarray(data[i]['a_upper']).astype(np.float32)), n)

    y = torch.from_numpy(np.asarray(data[i]['mean_MOCU']).astype(np.float32))
    y = y.unsqueeze(dim=0).unsqueeze(dim=0)

    data_ = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
    data_list.append(data_)

# train = data_list[0:28000]
# test = data_list[28000:30000]
#
# torch.save(train, '../Dataset/28000_7o_train.pth')
# torch.save(test, '../Dataset/2000_7o_test.pth')

train = data_list[0:70000]
test = data_list[70000:75000]

torch.save(train, '../Dataset/70000_5o_train.pth')
torch.save(test, '../Dataset/5000_5o_test.pth')