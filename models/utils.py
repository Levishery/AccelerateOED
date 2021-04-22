import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plotCurves(train_MSE, train_rank, test_MSE, EPOCH, name):
    exp_MSE = 0.00021
    print(f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: 0.000015794911;   data variance: "
          f"0.045936428010")
    exp_MSE = np.full(len(train_MSE), exp_MSE)
    plt.plot(train_MSE[1:], 'r', label="train_MSE")
    plt.plot(train_rank[1:], 'y', label="train_rank")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    name = name.split('.')[0]
    plt.savefig( '../Experiment/' + name + '/curve.png')

    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig('../Experiment/' + name + '/curve2.png')


def savePrediction(data, prediction, std, mean, name):
    result = np.zeros([len(data.y), 2])
    prediction = prediction * std + mean
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:len(data.y), 0]  # pre
    result[:, 1] = np.asarray([d * std + mean for d in data.y.cpu()])

    data = pd.DataFrame(result)

    writer = pd.ExcelWriter('../Experiment/' + name + '/Prediction.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.9f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()


def printInstance(data, prediction, std, mean):
    result = np.zeros([len(data.y), 2])
    prediction = prediction * std + mean
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:len(data.y), 0]  # pre
    result[:, 1] = np.asarray([d * std + mean for d in data.y.cpu()])
    print('w:')
    print(data.x[0])
    print('a_upper:')
    print(data.edge_attr[0][:, 0])
    print('a_lower:')
    print(data.edge_attr[0][:, 1])
    print('gt:')
    print(result[0, 1])
    print('prediction')
    print(prediction[0])