import time
from MOCU import *
import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader


def getEdgeAtt(attr1, attr2, N):
    edge_attr = torch.zeros([2, N * (N - 1)])
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr


def pre2R(attr, P_list, N):
    Matrix = np.zeros((N, N))
    k = 0
    for i in range(N):
        for j in range(i+1, N):
            Matrix[i, j] = attr[2*k]*P_list[k] + attr[2*k+1]*(1-P_list[k])
            k = k+1
    return Matrix


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


def findMPSequence(syncThresholds, isSynchronized, MOCUInitial, K_max, w, N, h, MVirtual, MReal, TVirtual, TReal,
                     aLowerBoundIn, aUpperBoundIn, it_idx, update_cnt, iterative=True):

    # load model
    print('loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().cuda()
    model.load_state_dict(torch.load('../model/MP_9and5_400.pth'))
    model.eval()
    # A = torch.zeros((10, 10), device="cuda:1")
    statistics = torch.load('../model/statistics_9and5_400.pth')
    mean = statistics['mean']
    std = statistics['std']

    edge_index = getEdgeAtt(np.tile(np.asarray([i for i in range(N)]), (N, 1)),
                            np.tile(np.asarray([[i] for i in range(N)]), (1, N)), N).long()
    x = torch.from_numpy(w.astype(np.float32))
    x = x.unsqueeze(dim=1)
    y = 0

    MOCUCurve = np.ones(update_cnt + 1) * 50.0
    MOCUCurve[0] = MOCUInitial
    timeComplexity = np.ones(update_cnt)

    aUpperBoundUpdated = aUpperBoundIn.copy()
    aLowerBoundUpdated = aLowerBoundIn.copy()

    optimalExperiments = []
    isInitiallyComputed = False
    R = np.zeros((N, N))
    R_ = np.zeros((N, N))
    for iteration in range(1, update_cnt + 1):
        iterationStartTime = time.time()
        if (not isInitiallyComputed) or iterative:
            # Computing the expected remaining MOCU
            # prepare data
            data_list = []
            P_syn_list = []
            for i in range(N):
                for j in range(i + 1, N):
                    isInitiallyComputed = True
                    aUpper = aUpperBoundUpdated.copy()
                    aLower = aLowerBoundUpdated.copy()

                    w_i = w[i]
                    w_j = w[j]
                    f_inv = 0.5 * np.abs(w_i - w_j)

                    aLower[j, i] = max(f_inv, aLower[i, j])
                    aLower[i, j] = aLower[j, i]

                    a_tilde = min(max(f_inv, aLowerBoundUpdated[i, j]), aUpperBoundUpdated[i, j])
                    P_syn = (aUpperBoundUpdated[i, j] - a_tilde) / (
                                aUpperBoundUpdated[i, j] - aLowerBoundUpdated[i, j])
                    P_syn_list.append(P_syn)

                    # ##
                    # it_temp_val = np.zeros(it_idx)
                    # for l in range(it_idx):
                    #     it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, T, aLower, aUpper, 0)
                    # MOCU_matrix_syn = np.mean(it_temp_val)
                    # ##

                    edge_attr = getEdgeAtt(torch.from_numpy(aLower.astype(np.float32)),
                                           torch.from_numpy(aUpper.astype(np.float32)), N)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
                    data_list.append(data)

                    aUpper = aUpperBoundUpdated.copy()
                    aLower = aLowerBoundUpdated.copy()

                    aUpper[i, j] = min(f_inv, aUpper[i, j])
                    aUpper[j, i] = aUpper[j, i]
                    it_temp_val = np.zeros(it_idx)

                    # ##
                    # for l in range(it_idx):
                    #     it_temp_val[l] = MOCU(K_max, w, N, h , MVirtual, T, aLower, aUpper, 0)
                    # MOCU_matrix_nonsyn = np.mean(it_temp_val)
                    # R_[i, j] = P_syn*MOCU_matrix_syn + (1-P_syn)*MOCU_matrix_nonsyn
                    # ##

                    edge_attr = getEdgeAtt(torch.from_numpy(aLower.astype(np.float32)),
                                           torch.from_numpy(aUpper.astype(np.float32)), N)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
                    data_list.append(data)
                else:
                    #
                    edge_attr = getEdgeAtt(torch.from_numpy(aLower.astype(np.float32)),
                                           torch.from_numpy(aUpper.astype(np.float32)), N)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
                    data_list.append(data)
            data = DataLoader(data_list, batch_size=128, shuffle=False)
            for d in data:
                d.to(device)
                prediction = model(d)
                prediction = prediction * std + mean
            prediction = prediction.cpu().detach().numpy()
            R = pre2R(prediction, P_syn_list, N)
            for i in range(N):
                for j in range(i + 1, N):
                    if (i, j) in optimalExperiments:
                        R[i, j] = 0
            print(R)
            # print(R_)

        min_ind = np.where(R == np.min(R[np.nonzero(R)]))

        if len(min_ind[0]) == 1:
            min_i_MOCU = int(min_ind[0])
            min_j_MOCU = int(min_ind[1])
        else:
            min_i_MOCU = int(min_ind[0][0])
            min_j_MOCU = int(min_ind[1][0])

        iterationTime = time.time() - iterationStartTime
        timeComplexity[iteration - 1] = iterationTime

        optimalExperiments.append((min_i_MOCU, min_j_MOCU))
        # print("selected experiment: ", min_i_MOCU, min_j_MOCU, "R: ", R[min_i_MOCU, min_j_MOCU])
        R[min_i_MOCU, min_j_MOCU] = 0.0
        f_inv = syncThresholds[min_i_MOCU, min_j_MOCU]

        if isSynchronized[min_i_MOCU, min_j_MOCU] == 0.0:
            aUpperBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aUpperBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
        else:
            aLowerBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aLowerBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)

        print("Iteration: ", iteration, ", selected: (", min_i_MOCU, min_j_MOCU, ")", iterationTime, "seconds")
        it_temp_val = np.zeros(it_idx)
        # for l in range(it_idx):
        #     it_temp_val[l] = MOCU(K_max, w, N, h, MReal, Torch, aLowerBoundUpdated, aUpperBoundUpdated, 0)
        # MOCUCurve[iteration] = np.mean(it_temp_val)
        print("before adjusting")
        print(MOCUCurve[iteration])
        if MOCUCurve[iteration] > MOCUCurve[iteration - 1]:
            MOCUCurve[iteration] = MOCUCurve[iteration - 1]
        print("The end of iteration: actual MOCU", MOCUCurve[iteration])
    print(optimalExperiments)
    return MOCUCurve, optimalExperiments, timeComplexity