import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import demjson
import random
from torch import nn
import argparse
import sys
import os

import numpy as np


class CNN(nn.Module):
    def __init__(self, N):
        super(CNN, self).__init__()
        if N == 7:
            self.nlayer = 3
        else:
            self.nlayer = 2
        # 5*5*3
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # 3*3*32
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        if self.nlayer == 3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3),
                nn.ReLU(inplace=True)
            )

        # 1*1*64
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        if self.nlayer == 3:
            x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden[0])  # hidden1 layer
        self.hidden2 = torch.nn.Linear(n_hidden[0], n_hidden[1])  # hidden2 layer
        self.predict = torch.nn.Linear(n_hidden[1], n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)  # linear output
        return x


def matrix2value(x):
    # 5*5 interaction matrix to [10,1] values
    x = np.tril(x, -1)
    x = x.ravel()[np.flatnonzero(x)]
    return x


def EdgeAtt2matrix(Attr, N):
    m = torch.zeros([N, N])
    k = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                m[i, j] = Attr[k]
                k = k + 1
    return m


def getArg():
    parser = argparse.ArgumentParser(description='Conv model')
    parser.add_argument('--data_path', default='',
                        help='')
    parser.add_argument('--model', default='CNN',
                        help='')
    parser.add_argument('--EPOCH', default=400, type=int,
                        help='EPOCH to train')
    parser.add_argument('--N', default=5, type=int,
                        help='System order')
    parser.add_argument('--test_only', action='store_true',
                        help='output test result only')
    parser.add_argument('--pretrain', default='.',
                        help='pretrained model name')
    parser.add_argument('--name', default='Conv',
                        help='file name to save the model')

    args = parser.parse_args()
    return args


def loadData(test_only, data_path, pretrain, name):

    print('Preparing data...')
    data_list = torch.load(data_path)

    if test_only:
        statistics = torch.load('../Experiment/' + pretrain + '/statistics.pth')
        mean = statistics['mean']
        std = statistics['std']
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = []
        data_test = data_list[0:len(data_list)]
    else:
        mean = np.asarray([d.y[0][0] for d in data_list]).mean()
        std = np.asarray([d.y[0][0] for d in data_list]).std()
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = data_list
        data_test = []
        torch.save({'mean': mean, 'std': std}, '../Experiment/' + name + '/statistics.pth')

    return data_train, data_test, [std, mean]


def main():

    print(sys.argv)
    args = getArg()
    global prediction
    if not os.path.exists('../Experiment/' + args.name):
        os.makedirs('../Experiment/' + args.name)
    N = args.N
    data_train, data_test, [std, mean] = loadData(args.test_only, args.data_path, args.pretrain, args.name)
    if args.test_only:
        data_ = data_test
    else:
        data_ = data_train

    if args.model == 'CNN':
        data_image = np.zeros([len(data_), 3, N, N])
        data_MOCU = np.zeros([len(data_), 1])
        for i in range(len(data_)):
            data_image[i, 0, :, :] = np.tile(np.asarray(data_[i].x.squeeze()), (N, 1))
            data_image[i, 1, :, :] = np.asarray(EdgeAtt2matrix(data_[i].edge_attr[:, 1], N), dtype=np.float64)
            data_image[i, 2, :, :] = np.asarray(EdgeAtt2matrix(data_[i].edge_attr[:, 0], N), dtype=np.float64)
            data_MOCU[i, 0] = data_[i].y
        model = CNN(N)
        data_x = data_image
    elif args.model == 'MLP':
        data_w = np.zeros([len(data_), N])
        data_a_upper = np.zeros([len(data_), int((N-1)/2*N)])
        data_a_lower = np.zeros([len(data_), int((N-1)/2*N)])
        data_MOCU = np.zeros([len(data_), 1])
        for i in range(len(data_)):
            data_w[i, :] = np.array(data_[i].x.squeeze())
            data_a_upper[i, :] = matrix2value(np.array(EdgeAtt2matrix(data_[i].edge_attr[:, 1], N), dtype=np.float64))
            data_a_lower[i, :] = matrix2value(np.array(EdgeAtt2matrix(data_[i].edge_attr[:, 0], N), dtype=np.float64))
            data_MOCU[i, 0] = data_[i].y
        data_x = np.concatenate((data_w, data_a_upper, data_a_lower), axis=1)
        model = Net(n_feature=N*N, n_hidden=[400, 200], n_output=1)
    else:
        raise ValueError('Only MLP and CNN')

    data_y = data_MOCU
    if args.test_only:  # train_x/y will not be used
        train_x = data_x
        test_x = data_x
        train_y = data_y
        test_y = data_y
    else:
        train_x = data_x[0:int(len(data_)*0.96), :]
        test_x = data_x[int(len(data_)*0.96):len(data_), :]
        train_y = data_y[0:int(len(data_)*0.96), :]
        test_y = data_y[int(len(data_)*0.96):len(data_), :]

    # numpy to tensor
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.float32)
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    print('Making model')

    model.cuda()
    if args.pretrain != '.':
        model.load_state_dict(torch.load('../Experiment/' + args.pretrain + '/model.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 256
    EPOCH = args.EPOCH

    test_MSE = np.zeros(EPOCH)
    train_MSE = np.zeros(EPOCH)

    torch_dataset = Data.TensorDataset(train_x, train_y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, )

    test_x = test_x.cuda()
    test_y = test_y.cuda()
    # test_y = test_y.unsqueeze(1)

    # start training
    if args.test_only:
        EPOCH = 1
    for epoch in range(EPOCH):
        if not args.test_only:
            train_MSE_step = []
            for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
                # train
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                prediction = model(batch_x)  # [batch_size, 1]
                # batch_y = batch_y.unsqueeze(1)  # Without Unsqueeze, PyTorch will broadcast [batch_size] to a wrong
                # tensor in the MSE loss

                loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)

                optimizer.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                train_MSE_step.append(loss.cpu())

            train_MSE[epoch] = sum(train_MSE_step) / len(train_MSE_step) * std * std
            print('epoch %d training MSE loss: %f' % (epoch, train_MSE[epoch]))

        # test
        prediction = model(test_x)
        loss = loss_func(prediction, test_y) * std * std
        print('epoch %d test MSE: %f' % (epoch, loss))
        test_MSE[epoch] = loss
        if epoch > 5 and loss < min(test_MSE[0:epoch]):
            torch.save(model.state_dict(), '../Experiment/' + args.name + '/model.pth')

    # exp1_y = np.asarray([d['MOCU1'] for d in data_[int(len(data_)*0.9):len(data_)]])
    # exp1_y = exp1_y.astype(np.float32)
    # exp1_y = torch.from_numpy(exp1_y)
    # exp1_y = exp1_y.unsqueeze(1).cuda()
    # exp2_y = np.asarray([d['MOCU2'] for d in data_[int(len(data_)*0.9):len(data_)]])
    # exp2_y = exp2_y.astype(np.float32)
    # exp2_y = torch.from_numpy(exp2_y)
    # exp2_y = exp2_y.unsqueeze(1).cuda()
    # exp_MSE = loss_func(exp2_y, exp1_y)
    exp_MSE = 0.0158*0.001
    var = std
    print(
        f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: {exp_MSE:.12f};   data variance: {var:.12f}")
    exp_MSE = np.full(len(train_MSE), exp_MSE)
    plt.plot(train_MSE[1:], 'r', label="train")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('../Experiment/' + args.name + '/train.png')

    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('../Experiment/' + args.name + '/train2.png')

    torch.save(model.state_dict(), '../Experiment/' + args.name + '/model.pth')

    # # save some prediction result
    # result = np.zeros([200, 7])
    # prediction = prediction.cpu().detach().numpy()
    # result[:, 0] = prediction[0:200, 0]  # pre
    # result[:, 1] = [d['MOCU1'] for d in data_[0:200]]  # exp1
    # result[:, 2] = [d['MOCU2'] for d in data_[0:200]]  # exp2
    #
    # data = pd.DataFrame(result)
    #
    # writer = pd.ExcelWriter('MLP_result.xlsx')  # 写入Excel文件
    # data.to_excel(writer, 'page_1', float_format='%.9f')  # ‘page_1’是写入excel的sheet名
    # writer.save()
    #
    # writer.close()

    # for i in range(200): print(f'w0: {test_x[i, 0]:.6f};    w1: {test_x[i, 1]:.6f};    a_upper: {test_x[i,
    # 2]:.6f};   a_lower: {test_x[i, 3]:.6f}') print(f'ground truth: {test_y[i, 0]:.6f};  prediction: {prediction[i,
    # 0]:.6f}')


if __name__ == '__main__':
    main()
