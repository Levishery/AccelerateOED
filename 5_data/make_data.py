import torch
import numpy as np
import demjson
import random


fileObject = open('data_5_oscillators_45000.json', 'r')
data_ = fileObject.read()
data_ = demjson.decode(data_)
fileObject = open('data_5_oscillators3.json', 'r')
data_1 = fileObject.read()
data_1 = demjson.decode(data_1)
fileObject = open('data_5_oscillators4.json', 'r')
data_2 = fileObject.read()
data_2 = demjson.decode(data_2)
data = data_+data_1+data_2

random.shuffle(data)
data_w = np.asarray([d['w'] for d in data])
data_w = torch.from_numpy(data_w.astype(np.float32))
data_a_upper = np.asarray([d['a_upper'] for d in data])
data_a_upper = torch.from_numpy(data_a_upper.astype(np.float32))
data_a_lower = np.asarray([d['a_lower'] for d in data])
data_a_lower = torch.from_numpy(data_a_lower.astype(np.float32))
data_MOCU = np.asarray([d['mean_MOCU'] for d in data])
data_MOCU = torch.from_numpy(data_MOCU.astype(np.float32))

dataset = {'w': data_w, 'data_a_upper': data_a_upper, 'data_a_lower': data_a_lower, 'MOCU': data_MOCU}
torch.save(dataset, 'TensorDataSet.pth')