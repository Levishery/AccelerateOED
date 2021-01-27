import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import torch.nn.functional as F


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


def find_MP_seq(reuse, save_f_inv, D_save, init_MOCU_val, K_max, w, N, h, M, T,
                a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt):
    model = Net().cuda()
    model.load_state_dict(torch.load('../5_data/MP_Graph.pth'))
    model.eval()
    MP_seq = np.ones(update_cnt + 1) * 50.0
    it_temp_val = np.zeros(it_idx)

    MP_seq[0] = init_MOCU_val

    Nnum = N * (N - 1) / 2
    i_set = np.zeros(update_cnt)
    j_set = np.zeros(update_cnt)
    ind_list = []
    for i in range(N):
        for j in range(i + 1, N):
            ind_list.append([[i, j]])
    random.shuffle(ind_list)

    for i in range(update_cnt):
        i_set[i] = np.asarray(ind_list[i])[0][0]
        j_set[i] = np.asarray(ind_list[i])[0][1]

    # print(i_set, j_set)
    for ij in range(1, update_cnt + 1):
        flag = 0

        i = int(i_set[ij - 1])
        j = int(j_set[ij - 1])

        f_inv = save_f_inv[i, j]

        if D_save[i, j] == 0.0:
            a_upper_bound_update[i, j] \
                = min(a_upper_bound_update[i, j], f_inv)
            a_upper_bound_update[j, i] \
                = min(a_upper_bound_update[i, j], f_inv)
            if f_inv > a_upper_bound_update[i, j]:
                flag = 1

        else:
            a_lower_bound_update[i, j] \
                = max(a_lower_bound_update[i, j], f_inv)
            a_lower_bound_update[j, i] \
                = max(a_lower_bound_update[i, j], f_inv)
            if f_inv < a_lower_bound_update[i, j]:
                flag = 1

        cnt = 0
        while Rand_seq[ij] > Rand_seq[ij - 1]:

            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update)

            Rand_seq[ij] = np.mean(it_temp_val)

            cnt = cnt + 1
            if cnt == 5:
                Rand_seq[ij] = Rand_seq[ij - 1]
                break

    return Rand_seq
