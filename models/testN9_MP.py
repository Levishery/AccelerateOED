import time
from MOCU import *
import random
import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import torch.multiprocessing as mp


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


def prediction(gpu, N, w, a_lower_bound, a_upper_bound):

    edge_index = getEdgeAtt(N, np.tile(np.asarray([i for i in range(N)]), (N, 1)),
                            np.tile(np.asarray([[i] for i in range(N)]), (1, N))).long()
    x = torch.from_numpy(w.astype(np.float32))
    x = x.unsqueeze(dim=1)
    edge_attr = getEdgeAtt(N, torch.from_numpy(a_lower_bound.astype(np.float32)),
                           torch.from_numpy(a_upper_bound.astype(np.float32)))
    y = 0
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
    data = DataLoader([data], batch_size=128, shuffle=False)
    statistics = torch.load('data/statistics_9and5_400.pth')
    mean = statistics['mean']
    std = statistics['std']

    # load model
    device = torch.device("cuda:" + str(gpu))
    model = Net().cuda()
    model.load_state_dict(torch.load('data/MP_9and5_400.pth'))
    model.eval()

    for d in data:
        d.to(device)
        prediction = model(d)
        prediction = prediction * std + mean

    print(prediction)


if __name__ == '__main__':
    # get random data
    N = 9

    w = np.zeros(N)
    # for i in range(N):
    #     w[i] = 12 * (0.5 - random.random())
    #
    # a_upper_bound = np.zeros((N, N))
    # a_lower_bound = np.zeros((N, N))
    # uncertainty = 0.3 * random.random()
    # for i in range(N):
    #     if random.random() < 0.5:
    #         mul_ = 0.6
    #     else:
    #         mul_ = 1.1
    #     for j in range(i + 1, N):
    #         mul = mul_ * random.random()
    #         f_inv = np.abs(w[i] - w[j]) / 2.0
    #         a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
    #         a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
    #         a_upper_bound[j, i] = a_upper_bound[i, j]
    #         a_lower_bound[j, i] = a_lower_bound[i, j]

    w[0] = 1.19
    w[1] = 3.23
    w[2] = 6.34
    w[3] = 7.48
    w[4] = 10.9
    w[5] = 11.62
    w[6] = 14.74
    w[7] = 29.58
    w[8] = 38.88

    aInitialUpper = np.zeros((N,N))
    aInitialLower = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            if (i <= 4) and (j <= 4):
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.03
                aInitialLower[i, j] = f_inv*0.97
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]
            elif (i >= 7) or (j >= 7):
                f_inv = np.abs(w[i] - w[j])/9.0
                aInitialUpper[i, j] = f_inv*1.02
                aInitialLower[i, j] = f_inv*0.98
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]
            else:
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.01
                aInitialLower[i, j] = f_inv*0.99
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]

            if j == 5:
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.02
                aInitialLower[i, j] = f_inv*0.98
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]

    a_upper_bound = aInitialUpper
    a_lower_bound = aInitialLower
    a_upper_bound_update = a_upper_bound.copy()
    a_lower_bound_update = a_lower_bound.copy()

    context = mp.spawn(prediction, nprocs=1, join=False, args=(N, w, a_lower_bound, a_upper_bound))
    context.join()

    K_max = 20000  # 5000

    # Final Time
    T = 4.0

    # Time discretization
    h = 1.0 / 160.0
    M = int(T / h)

    MOCU_val = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0)
    print(MOCU_val)