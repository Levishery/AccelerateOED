import sys

sys.path.append("./src")

from find_MOCU_seq import *
from find_Entropy_seq import *
from find_Rand_seq import *
from MOCU import *
from main_module_ij_couple import *
import time

import numpy as np
import random
import matplotlib.pyplot as plt

print("################################################")
print("Test code")
print("\n")

print("Start ticking")
tt = time.time()

it_idx = 1
update_cnt = 10
# Number of equations
N = 2
K_max = 20000  # 5000

# Final Time
T = 4.0

# Time discretization
h = 1.0 / 160.0

# Time steps
M = int(T / h)
w = np.zeros(N)
step = 20000
data_matrix = np.zeros([step, 8])
for s in range(step):
    w[0] = 12 * (0.5 - random.random())
    w[1] = 12 * (0.5 - random.random())
    data_matrix[s, 0] = w[0]
    data_matrix[s, 1] = w[1]

    print('w0: %f, w1: %f' % (w[0], w[1]))

    a_upper_bound = np.zeros((N, N))
    a_lower_bound = np.zeros((N, N))
    a_lower_bound_update = np.zeros((N, N))
    a_upper_bound_update = np.zeros((N, N))
    a = np.zeros((N, N))
    tata = np.zeros((N, N))

    uncertainty = 0.3 * random.random()
    for i in range(N):
        for j in range(i + 1, N):
            f_inv = np.abs(w[i] - w[j]) / 2.0
            a_upper_bound[i, j] = f_inv * (1 + uncertainty)
            a_lower_bound[i, j] = f_inv * (1 - uncertainty)
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]

    mul = 1.3 * random.random()
    a_upper_bound[0, 1] = a_upper_bound[0, 1] * mul
    a_lower_bound[0, 1] = a_lower_bound[0, 1] * mul
    data_matrix[s, 2] = a_upper_bound[0, 1]
    data_matrix[s, 3] = a_lower_bound[0, 1]
    for i in range(N):
        for j in range(i + 1, N):
            a_upper_bound[j, i] = a_upper_bound[i, j]
            a_lower_bound[j, i] = a_lower_bound[i, j]

    a_lower_bound_update = a_lower_bound.copy()
    a_upper_bound_update = a_upper_bound.copy()

    print('upper: %f, lower: %f' % (a_upper_bound[0, 1], a_lower_bound[0, 1]))

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

    MOCU_val1 = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update)

    print('MOCU value 1: %f' % MOCU_val1)
    data_matrix[s, 4] = MOCU_val1

    MOCU_val2 = MOCU(K_max, w, N, h, M, T, a_lower_bound_update, a_upper_bound_update)

    print('MOCU value 2: %f' % MOCU_val2)
    data_matrix[s, 5] = MOCU_val2

    data_matrix[s, 6] = abs(MOCU_val2 - MOCU_val1)
    data_matrix[s, 7] = (MOCU_val2 + MOCU_val1)/2

np.save("data_matrix.npy", data_matrix)