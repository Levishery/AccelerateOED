import sys

sys.path.append("./src")

from findMPSequence import getMPSequence, findMPSequence
import matplotlib.pyplot as plt
from MOCU import *
from mocu_comp import *
import json

import random
import torch.multiprocessing as mp

import time
import numpy as np
from determineSyncTwo import *

if __name__ == '__main__':
    print("################################################")
    print("Test code")
    print("\n")

    print("Start ticking")
    tt = time.time()

    # Number of equations
    # N = 5
    N = 7
    K_max = 20480  # 5000

    # Final Time
    T = 5.0

    # Time discretization
    h = 1.0 / 160.0

    # Time steps
    M = int(T / h)
    w = np.zeros(N)
    step = 20
    data_ = []
    for s in range(step):
        data_dic = {}
        for i in range(N):
            w[i] = 20 * (0.5 - random.random())

        data_dic['w'] = w.tolist()
        print(w, '<- w')

        a_upper_bound = np.zeros((N, N))
        a_lower_bound = np.zeros((N, N))
        a_lower_bound_update = np.zeros((N, N))
        a_upper_bound_update = np.zeros((N, N))
        a = np.zeros((N, N))
        tata = np.zeros((N, N))

        for i in range(N):
            if random.random() < 0.5: # case 1
                mul_ = 0.35
            else:
                mul_ = 1.2
            for j in range(i + 1, N):
                # if random.random() < 0.5: # case 2
                #     mul_ = 0.35
                # else:
                #     mul_ = 1.2
                mul = mul_ * random.random()
                uncertainty = 0.6 * random.random()
                f_inv = np.abs(w[i] - w[j]) / 2.0
                a_upper_bound[i, j] = f_inv * (1 + uncertainty) * mul
                a_lower_bound[i, j] = f_inv * (1 - uncertainty) * mul
                a_upper_bound[j, i] = a_upper_bound[i, j]
                a_lower_bound[j, i] = a_lower_bound[i, j]

        data_dic['a_upper'] = a_upper_bound.tolist()
        data_dic['a_lower'] = a_lower_bound.tolist()

        aInitialLower = a_lower_bound.copy()
        aInitialUpper = a_upper_bound.copy()

        w = np.loadtxt('../7o_data/w6.txt')
        aInitialUpper = np.loadtxt('../7o_data/upper6.txt')
        aInitialLower = np.loadtxt('../7o_data/lower6.txt')

        print(a_upper_bound, '<- a_upper')
        print(a_lower_bound, '<- a_lower')


        it_idx = 3
        update_cnt = 5
        N = 7
        K_max = 20480

        # Time must be larger than 250 for N = 5
        deltaT = 1.0 / 160.0
        TVirtual = 5
        MVirtual = int(TVirtual / deltaT)
        TReal = 5
        MReal = int(TReal / deltaT)
        it_temp_val = np.zeros(it_idx)

        isSynchronized = np.zeros((N, N))
        criticalK = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1,N):
                randomNumber = random.random()
                a[i,j] = aInitialLower[i,j] + randomNumber*(aInitialUpper[i,j] - aInitialLower[i,j])
                a[j,i] = a[i,j]

        for i in range(N):
            for j in range(i + 1, N):
                w_i = w[i]
                w_j = w[j]
                a_ij = a[i, j]
                syncThreshold = 0.5 * np.abs(w_i - w_j)
                criticalK[i, j] = syncThreshold
                criticalK[j, i] = syncThreshold
                isSynchronized[i, j] = determineSyncTwo(w_i, w_j, deltaT, 2, MReal, a_ij)

        for l in range(it_idx):
            it_temp_val[l] = MOCU(K_max, w, N, deltaT, MReal, TReal, aInitialLower.copy(), aInitialUpper.copy(), 0)
        MOCUInitial = np.mean(it_temp_val)
        print(" Initial MOCU: ", MOCUInitial)
        if MOCUInitial < 0.5:
            continue
        aUpperUpdated = aInitialUpper.copy()
        aLowerUpdated = aInitialLower.copy()

        iterative = True
        smp = mp.get_context('spawn')
        q = smp.SimpleQueue()
        context = mp.spawn(getMPSequence, nprocs=1, join=False, args=(q, criticalK, isSynchronized,
                                                                      w, N, aLowerUpdated, aUpperUpdated,
                                                                      update_cnt, iterative))
        context.join()
        [optimalExperiments, update_bond, timeComplexity] = q.get()

        MOCUCurve, experimentSequence, timeComplexity = findMPSequence(optimalExperiments, update_bond,
                                                                       timeComplexity, MOCUInitial, K_max, w, N,
                                                                       deltaT, MVirtual, MReal, TVirtual, TReal,
                                                                       aLowerUpdated, aUpperUpdated, it_idx,
                                                                       update_cnt)

        plt.plot(MOCUCurve)
        plt.show()

        # jsObj = json.dumps(data_dic)
        # fileObject = open('../Dataset/7o_simulation' + str(s) + '.json', 'w')
        # fileObject.write(jsObj)
        # fileObject.close()

        # np.savetxt('../7o_data/lower6.txt', aInitialLower, fmt='%.64e')
        # np.savetxt('../7o_data/upper6.txt', aInitialUpper, fmt='%.64e')
        # np.savetxt('../7o_data/w6.txt', w, fmt='%.64e')