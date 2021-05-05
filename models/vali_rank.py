from N7ForShare.MOCU import *
import random
import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

def getEdgeAtt(N, attr1, attr2):
    edge_attr = torch.zeros([2, N * (N - 1)])
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr

def EdgeAtt2matrix(Attr, N):
    m = torch.zeros([N, N])
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                m[i, j] = Attr[k]
                k = k + 1
    return m


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 32
        self.lin0 = torch.nn.Linear(1, dim)

        nn = Sequential(Linear(2, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


def prediction(N, w, a_lower_bound, a_upper_bound, model):

    edge_index = getEdgeAtt(N, np.tile(np.asarray([i for i in range(N)]), (N, 1)),
                            np.tile(np.asarray([[i] for i in range(N)]), (1, N))).long()
    x = torch.from_numpy(w.astype(np.float32))
    x = x.unsqueeze(dim=1)
    edge_attr = getEdgeAtt(N, torch.from_numpy(a_lower_bound.astype(np.float32)),
                           torch.from_numpy(a_upper_bound.astype(np.float32)))
    y = 0
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
    data = DataLoader([data], batch_size=128, shuffle=False)


    for d in data:
        d.to(device)
        prediction = model(d)

    return prediction


if __name__ == '__main__':
    K_max = 20480  # 5000

    # Final Time
    T = 5.0

    # Time discretization
    h = 1.0 / 160.0
    M = int(T / h)
    # get random data
    # N = 5
    is_ODE = 1
    N = 7
    up_flag = 0
    down_flag = 0
    data_ = []
    w = np.zeros(N)
    # load model
    if not is_ODE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().cuda()
        # model.load_state_dict(torch.load('../Experiment/noconsmixed_/model.pth'))
        model.load_state_dict(torch.load('../Experiment/consmixed_/model.pth'))
        model.eval()

    if N == 7:
        data_list = torch.load('../Dataset/2000_7o_test.pth')
    else:
        data_list = torch.load('../Dataset/3000_5o_test.pth')
    step = len(data_list)
    for d in tqdm(data_list):
        data_dic = {}
        # data = DataLoader([d], batch_size=128, shuffle=False)
        # for x in data:
        #     x.to(device)
        #     pre = model(x)
        w = np.asarray(d.x.squeeze())
        a_upper_bound = np.asarray(EdgeAtt2matrix(d.edge_attr[:, 1], N))
        a_lower_bound = np.asarray(EdgeAtt2matrix(d.edge_attr[:, 0], N))
        # for i in range(N):
        #     if N == 5:
        #         w[i] = 12 * (0.5 - random.random())
        #     else:
        #         w[i] = 20 * (0.5 - random.random())
        #
        # data_dic['w'] = w.tolist()
        #
        # a_upper_bound = np.zeros((N, N))
        # a_lower_bound = np.zeros((N, N))
        # a_lower_bound_update = np.zeros((N, N))
        # a_upper_bound_update = np.zeros((N, N))
        # a = np.zeros((N, N))
        #
        # if N == 5:
        #     uncertainty = 0.3 * random.random()
        # else:
        #     uncertainty = 0.6 * random.random()
        # for i in range(N):
        #     if N == 5:
        #         if random.random() < 0.5:
        #             mul_ = 0.6
        #         else:
        #             mul_ = 1.1
        #     else:
        #         if random.random() < 0.5:  # case 1
        #             mul_ = 0.35
        #         else:
        #             mul_ = 1.2
        #     for j in range(i + 1, N):
        #         # if random.random() < 0.5: # case 2
        #         #     mul_ = 0.35
        #         # else:
        #         #     mul_ = 1.2
        #         mul = mul_ * random.random()
        #         f_inv = np.abs(w[i] - w[j]) / 2.0
        #         a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
        #         a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
        #         a_upper_bound[j, i] = a_upper_bound[i, j]
        #         a_lower_bound[j, i] = a_lower_bound[i, j]
        if is_ODE:
            pre = (MOCU(K_max, w, N, h, M, T, a_lower_bound.copy(), a_upper_bound.copy(), 0) +
                   MOCU(K_max, w, N, h, M, T, a_lower_bound.copy(), a_upper_bound.copy(), 0))/2
        else:
            pre = prediction(N, w, a_lower_bound, a_upper_bound, model)
        a_lower_bound_update = a_lower_bound.copy()
        a_upper_bound_update = a_upper_bound.copy()

        i = int(np.floor(N*random.random()))
        j = int(np.floor((N-1)*random.random()))
        if j >= i:
            j = j+1
        a_upper_bound_update[i, j] = (a_lower_bound[i, j] + a_upper_bound[i, j])/2
        a_upper_bound_update[j, i] = a_upper_bound_update[i, j]
        if is_ODE:
            pre3 = (MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0) +
                    MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0))/2
        else:
            pre3 = prediction(N, w, a_lower_bound_update, a_upper_bound_update, model)
        if pre3<pre:
            down_flag = down_flag + 1

        a_lower_bound_update = a_lower_bound.copy()
        a_upper_bound_update = a_upper_bound.copy()
        i = int(np.floor(N * random.random()))
        j = int(np.floor((N-1) * random.random()))
        if j >= i:
            j = j + 1
        a_lower_bound_update[i, j] = (a_lower_bound[i, j] + a_upper_bound[i, j]) / 2
        a_lower_bound_update[j, i] = a_lower_bound_update[i, j]
        if is_ODE:
            pre2 = (MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0) +
                    MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0))/2
        else:
            pre2 = prediction(N, w, a_lower_bound_update, a_upper_bound_update, model)
        if pre2 < pre:
            up_flag = up_flag + 1

    print(up_flag)
    print(down_flag)
    print(up_flag/step)
    print(down_flag/step)






