import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import demjson
import random

import numpy as np

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

def main():
    global prediction
    print('Preparing data...')
    # fileObject = open('data_5_oscillators_45000.json', 'r')
    fileObject = open('data_5_oscillators.json', 'r')
    data_ = fileObject.read()
    data_ = demjson.decode(data_)
    random.shuffle(data_)
    data_w = np.zeros([len(data_), 5])
    data_a_upper = np.zeros([len(data_), 10])
    data_a_lower = np.zeros([len(data_), 10])
    data_MOCU = np.zeros([len(data_), 1])
    for i in range(len(data_)):
        data_w[i, :] = np.array(data_[i]['w'])
        data_a_upper[i, :] = matrix2value(np.array(data_[i]['a_upper']))
        data_a_lower[i, :] = matrix2value(np.array(data_[i]['a_lower']))
        data_MOCU[i, 0] = data_[i]['mean_MOCU']

    data_x = np.concatenate((data_w, data_a_upper, data_a_lower), axis=1)
    data_y = data_MOCU
    train_x = data_x[0:int(len(data_)*0.9), :]
    test_x = data_x[int(len(data_)*0.9):len(data_), :]
    train_y = data_y[0:int(len(data_)*0.9), :]
    test_y = data_y[int(len(data_)*0.9):len(data_), :]

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
    # model = Net(n_feature=4, n_hidden=[400, 200], n_output=1)
    model = Net(n_feature=25, n_hidden=[400, 200], n_output=1)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    BATCH_SIZE = 256
    EPOCH = 400

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
    for epoch in range(EPOCH):
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

        train_MSE[epoch] = sum(train_MSE_step) / len(train_MSE_step)
        print('epoch %d training MSE loss: %f' % (epoch, train_MSE[epoch]))

        # test
        prediction = model(test_x)
        loss = loss_func(prediction, test_y)
        print('epoch %d test MSE: %f' % (epoch, loss))
        test_MSE[epoch] = loss

    exp1_y = np.asarray([d['MOCU1'] for d in data_[int(len(data_)*0.9):len(data_)]])
    exp1_y = exp1_y.astype(np.float32)
    exp1_y = torch.from_numpy(exp1_y)
    exp1_y = exp1_y.unsqueeze(1).cuda()
    exp2_y = np.asarray([d['MOCU2'] for d in data_[int(len(data_)*0.9):len(data_)]])
    exp2_y = exp2_y.astype(np.float32)
    exp2_y = torch.from_numpy(exp2_y)
    exp2_y = exp2_y.unsqueeze(1).cuda()
    exp_MSE = loss_func(exp2_y, exp1_y)
    var = np.var(test_y.cpu().detach().numpy())
    print(
        f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: {exp_MSE.cpu():.12f};   data variance: {var:.12f}")
    exp_MSE = np.full(len(train_MSE), exp_MSE.cpu())
    plt.plot(train_MSE[1:], 'r', label="train")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('train_MLP.png')

    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('train_MLP2.png')

    torch.save(model.state_dict(), '/media/public/likang/chengqihua/projects/MOCU/Simple_pair/MLP.pth')

    # save some prediction result
    result = np.zeros([200, 7])
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:200, 0]  # pre
    result[:, 1] = [d['MOCU1'] for d in data_[0:200]]  # exp1
    result[:, 2] = [d['MOCU2'] for d in data_[0:200]]  # exp2

    data = pd.DataFrame(result)

    writer = pd.ExcelWriter('MLP_result.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.9f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()

    # for i in range(200): print(f'w0: {test_x[i, 0]:.6f};    w1: {test_x[i, 1]:.6f};    a_upper: {test_x[i,
    # 2]:.6f};   a_lower: {test_x[i, 3]:.6f}') print(f'ground truth: {test_y[i, 0]:.6f};  prediction: {prediction[i,
    # 0]:.6f}')


if __name__ == '__main__':
    main()
