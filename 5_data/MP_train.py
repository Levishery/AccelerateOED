from utils import *
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, DataLoader
import numpy as np

import argparse

N = 5


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


def getEdgeAtt(attr1, attr2):
    edge_attr = torch.zeros([2, N * (N - 1)])
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                edge_attr[0, k] = attr1[i, j]
                edge_attr[1, k] = attr2[i, j]
                k = k + 1
    return edge_attr


def getArg():
    parser = argparse.ArgumentParser(description='Message Passing for MOCU Prediction')
    parser.add_argument('--pretrain', default='.',
                        help='pretrained model')
    parser.add_argument('--save_model', default='MP_Graph.pth',
                        help='file name to save the model')
    parser.add_argument('--data_path', default='',
                        help='')
    parser.add_argument('--EPOCH', default=200,
                        help='EPOCH to train')
    parser.add_argument('--test_only', action='store_true',
                        help='output test result only')
    parser.add_argument('--debug', action='store_true',
                        help='print debug information')
    args = parser.parse_args()
    return args


def loadData(test_only, data_path, pretrain):
    print('Preparing data...')
    # fileObject = open('data_5_oscillators_45000.json', 'r')
    if 'Graph' in data_path:
        data_list = torch.load(data_path)
    else:
        dataSet = torch.load(data_path)
        data_w = dataSet['w']
        data_a_upper = dataSet['data_a_upper']
        data_a_lower = dataSet['data_a_lower']
        data_MOCU = dataSet['MOCU']

        data_list = []
        # fully connected
        edge_index = getEdgeAtt(np.tile(np.asarray([i for i in range(N)]), (N, 1)),
                                np.tile(np.asarray([[i] for i in range(N)]), (1, N))).long()
        for i in range(len(data_w)):
            x = data_w[i]
            x = x.unsqueeze(dim=1)
            edge_attr = getEdgeAtt(data_a_lower[i], data_a_upper[i])
            y = data_MOCU[i].unsqueeze(dim=0).unsqueeze(dim=0)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.t(), y=y)
            data_list.append(data)

        torch.save(data_list, data_path.replace('Tensor', 'Graph'))
    # data.x: Node feature matrix with shape[num_nodes, num_node_features]
    #
    # data.edge_index: Graph connectivity in COO format with shape[2, num_edges] and type torch.long
    #
    # data.edge_attr: Edge feature matrix with shape[num_edges, num_edge_features]
    #
    # data.y: Target to train against(may have arbitrary shape), e.g., node - level targets of shape[num_nodes, *]
    # or graph - level targets of shape[1, *]
    # Normalize targets to mean = 0 and std = 1.
    if test_only:
        statistics = torch.load(pretrain.replace('MP', 'statistics'))
        mean = statistics['mean']
        std = statistics['std']
        for d in data_list:
            d.y = (d.y - mean) / std
        data_test = data_list[0:len(data_list)]
        # data_test = [d for d in data_list if d.num_nodes == 5]
        train_loader = []
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)
    else:
        mean = np.asarray([d.y for d in data_list]).mean()
        std = np.asarray([d.y for d in data_list]).std()
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = data_list[0:int(0.96 * len(data_list))]
        data_test = data_list[int(0.96 * len(data_list)):len(data_list)]
        train_loader = DataLoader(data_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)

    return train_loader, test_loader, [std, mean]


def main():

    args = getArg()
    EPOCH = args.EPOCH if not args.test_only else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, [std, mean] = loadData(args.test_only, args.data_path, args.pretrain)

    print('Making Model...')
    model = Net().cuda()
    if args.pretrain != '.':
        model.load_state_dict(torch.load(args.pretrain))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5, min_lr=0.00001)
    test_MSE = np.zeros(EPOCH)
    train_MSE = np.zeros(EPOCH)
    # start training
    for epoch in range(EPOCH):
        if not args.test_only:
            train_MSE_step = []
            model.train()
            for data in train_loader:  # for each training step
                # train
                lr = scheduler.optimizer.param_groups[0]['lr']
                data = data.to(device)
                optimizer.zero_grad()
                prediction = model(data).unsqueeze(dim=1)  # [batch_size, 1]
                loss = F.mse_loss(prediction, data.y)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                train_MSE_step.append(loss.item() * std * std)

            train_MSE[epoch] = (sum(train_MSE_step) / len(train_MSE_step))
            print('epoch %d learning rate %f training MSE loss: %f' % (epoch, lr, train_MSE[epoch]))

        # test
        model.eval()
        error = 0
        for data in test_loader:
            data = data.to(device)
            prediction = model(data).unsqueeze(dim=1)
            error += (prediction * std - data.y * std).square().sum().item()  # MSE
        loss = error / len(test_loader.dataset)
        print('epoch %d test MSE: %f' % (epoch, loss))
        print('std:%f, mean:%f' %(std, mean))
        test_MSE[epoch] = loss
        if epoch > 5 and loss < min(test_MSE[0:epoch]):
            torch.save(model.state_dict(), args.save_model)

    if not args.test_only:
        torch.save({'mean': mean, 'std': std}, args.save_model.replace('MP', 'statistics'))
    # plot and save
    plotCurves(train_MSE, test_MSE, EPOCH)

    # save some prediction result
    savePrediction(data, prediction, std, mean)

    if args.debug:
        printInstance(data, prediction, std, mean)

if __name__ == '__main__':
    main()
