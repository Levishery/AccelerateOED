import sys

sys.path.append("./src")

from MOCU import *
from mocu_comp import *
import time
import json

import numpy as np
import random

print("################################################")
print("Test code")
print("\n")

print("Start ticking")
tt = time.time()

# Number of equations
N = 5
K_max = 20480  # 5000

# Final Time
T = 4.0

# Time discretization
h = 1.0 / 160.0

# Time steps
M = int(T / h)
w = np.zeros(N)
step = 50000
data_ = []
for s in range(step):
    data_dic = {}
    for i in range(N):
        w[i] = 12 * (0.5 - random.random())

    data_dic['w'] = w.tolist()
    print(w, '<- w')

    a_upper_bound = np.zeros((N, N))
    a_lower_bound = np.zeros((N, N))
    a_lower_bound_update = np.zeros((N, N))
    a_upper_bound_update = np.zeros((N, N))
    a = np.zeros((N, N))
    tata = np.zeros((N, N))

    uncertainty = 0.3 * random.random()
    for i in range(N):
        if random.random() < 0.5:
            mul_ = 0.6
        else:
            mul_ = 1.1
        for j in range(i + 1, N):
            # if random.random() < 0.5:
            #     mul_ = 0.6
            # else:
            #     mul_ = 1.1
            mul = mul_ * random.random()
            f_inv = np.abs(w[i] - w[j]) / 2.0
            a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
            a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]

    data_dic['a_upper'] = a_upper_bound.tolist()
    data_dic['a_lower'] = a_lower_bound.tolist()

    a_lower_bound_update = a_lower_bound.copy()
    a_upper_bound_update = a_upper_bound.copy()

    print(a_upper_bound, '<- a_upper')
    print(a_lower_bound, '<- a_lower')

    # check the mean a
    for i in range(N):
        for j in range(i + 1, N):
            a[i, j] = a_lower_bound[i, j] + 1 / 2 * (a_upper_bound[i, j] - a_lower_bound[i, j])
            a[j, i] = a[i, j]

    init_sync_check = mocu_comp(w, h, N, M, a)

    if init_sync_check == 1:
        print('It is already sync system with mean a!!!!')

    for i in range(N):
        for j in range(i + 1, N):
            a[i, j] = a_lower_bound[i, j]
            a[j, i] = a[i, j]

    init_sync_check = mocu_comp(w, h, N, M, a)

    if init_sync_check == 1:
        print('It is already sync system with the lower bound!!!!')

    MOCU_val1 = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0)

    print('MOCU value 1: %f' % MOCU_val1)
    data_dic['MOCU1'] =  MOCU_val1

    MOCU_val2 = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update, 0)

    print('MOCU value 2: %f' % MOCU_val2)
    data_dic['MOCU2'] =  MOCU_val2

    data_dic['mean_MOCU'] = (MOCU_val2 + MOCU_val1)/2

    data_.append(data_dic)
    print(s)

# data_.tolist()

print(type(data_))
jsObj = json.dumps(data_)

fileObject = open('../Dataset/5o_type1.json', 'w')
fileObject.write(jsObj)
fileObject.close()