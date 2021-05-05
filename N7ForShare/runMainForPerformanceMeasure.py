# Youngjoon Hong (07.15.2020)
# !!!!!!!!!!!!!!!!!!!!Important!!!!!!!!!!!!!!!!!!!!!!!
# If you want to chage the number of oscillators, Change "N" here.
# In addition, please do not forget update "manually" the followings in the MOCU.py file
# #define N_global 10. This is much faster than passing the params in pyCUDA.

import sys
import time

sys.path.append("./src")

from findMOCUSequence import *
from findEntropySequence  import *
from findRandomSequence import *
from findMPSequence import *
from determineSyncTwo import *
from determineSyncN import *
import torch.multiprocessing as mp

import numpy as np

if __name__ == '__main__':
    it_idx = 5
    update_cnt = 10
    N = 7
    K_max = 20480

    # Time must be larger than 250 for N = 5
    deltaT = 1.0/160.0
    TVirtual = 5
    MVirtual = int(TVirtual/deltaT)
    TReal = 5
    MReal = int(TReal/deltaT)

    # w = np.zeros(N)
    # w[0] = -3.4600006884189014044750365428626537322998046875000000000000000000e+00
    # w[1] = -1.9611565409805971071932617633137851953506469726562500000000000000e+00
    # w[2] = -6.7543611126664915289552482136059552431106567382812500000000000000e-01
    # w[3] = -3.8065462701773888909428933402523398399353027343750000000000000000e-01
    # w[4] = -3.6750415163579663868631541845388710498809814453125000000000000000e-01
    # w[5] = 6.1160931647836296320974724949337542057037353515625000000000000000e+00
    # w[6] = 8.3287422947517093518854380818083882331848144531250000000000000000e+00
    w = np.asarray([-2.80959771, -7.89198094, 5.37590268, -4.40259349, -8.73997011,  7.87388523, 2.96707722])

    listMethods = ['iMP', 'MP', 'ODE', 'RANDOM', 'ENTROPY']
    numberOfSimulationsPerMethod = 1
    numberOfVaildSimulations = 0
    numberOfSimulations = 0
    save_MOCU_matrix = np.zeros([update_cnt + 1, len(listMethods), numberOfSimulationsPerMethod])

    aInitialUpper = np.loadtxt('../7o_data/upper2.txt')
    aInitialLower = np.loadtxt('../7o_data/lower2.txt')

    while (numberOfSimulationsPerMethod > numberOfVaildSimulations):

        randomState = np.random.RandomState(int(numberOfSimulations))
        a = np.zeros((N,N))
        for i in range(N):
            for j in range(i+1,N):
                randomNumber = randomState.uniform()
                a[i,j] = aInitialLower[i,j] + randomNumber*(aInitialUpper[i,j] - aInitialLower[i,j])
                a[j,i] = a[i,j]

        # a = np.loadtxt('7o_data/couplingStrength.txt')

        numberOfSimulations += 1

        # #test
        # for i in range(N):
        #     for j in range(i+1,N):
        #         a[i,j] = aInitialUpper[i, j]
        #         a[j,i] = a[i,j]
        # #test

        init_sync_check = determineSyncN(w, deltaT, N, MReal, a)

        if init_sync_check == 1:
            print('             The system has been already stable.')
            continue
        else:
            print('             Unstable system has been found')

        # print("initial a")
        # print(a)
        # print("aUpperBoundInitial")
        # print(aInitialUpper)
        # print("aLowerBoundInitial")
        # print(aInitialLower)

        isSynchronized = np.zeros((N,N))
        criticalK = np.zeros((N,N))

        for i in range(N):
            for j in range(i+1,N):
                w_i = w[i]
                w_j = w[j]
                a_ij = a[i,j]
                syncThreshold = 0.5*np.abs(w_i - w_j)
                criticalK[i, j] = syncThreshold
                criticalK[j, i] = syncThreshold
                isSynchronized[i,j] = determineSyncTwo(w_i, w_j, deltaT, 2, MReal, a_ij)

        # print("criticalK")
        # print(criticalK)
        # print("isSynchronized")
        # print(isSynchronized)

        # if (len(isSynchronized[np.nonzero(isSynchronized)]) < N/2) or ((N*(N-1)/2) - len(isSynchronized[np.nonzero(isSynchronized)]) < N/2):
        #     print('                     Not favorable system')
        #     continue
        # else:
        #     print('                     Favorable system has been found')

        np.savetxt('../results/paramCouplingStrength' + str(numberOfVaildSimulations) + '.txt', a, fmt='%.64e')
        for indexMethod in range(len(listMethods)):
            timeMOCU = time.time()
            it_temp_val = np.zeros(it_idx)
            for l in range(it_idx):
                it_temp_val[l] = MOCU(K_max, w, N, deltaT, MReal, TReal, aInitialLower.copy(), aInitialUpper.copy(), 0)
            MOCUInitial  = np.mean(it_temp_val)
            print("Round: ", numberOfVaildSimulations, "/", numberOfSimulationsPerMethod, "-", listMethods[indexMethod],
                  "Iteration: ", numberOfVaildSimulations, " Initial MOCU: ", MOCUInitial, " Computation time: ",
                  time.time() - timeMOCU)
            aUpperUpdated = aInitialUpper.copy()
            aLowerUpdated = aInitialLower.copy()

            if listMethods[indexMethod] == 'RANDOM':
                MOCUCurve, experimentSequence, timeComplexity = findRandomSequence(criticalK, isSynchronized,
                                                                                   MOCUInitial, K_max, w, N, deltaT,
                                                                                   MReal, TReal, aLowerUpdated,
                                                                                   aUpperUpdated, it_idx, update_cnt)

            elif listMethods[indexMethod] == 'ENTROPY':
                MOCUCurve, experimentSequence, timeComplexity = findEntropySequence(criticalK, isSynchronized,
                                                                                    MOCUInitial, K_max, w, N, deltaT,
                                                                                    MReal, TReal, aLowerUpdated,
                                                                                    aUpperUpdated, it_idx, update_cnt)

            elif listMethods[indexMethod] == 'iODE':
                iterative = True
                print("iterative: ", iterative)
                MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial,
                                                                                 K_max, w, N, deltaT, MVirtual, MReal,
                                                                                 TVirtual, TReal, aLowerUpdated,
                                                                                 aUpperUpdated, it_idx,
                                                                                 update_cnt, iterative=iterative)

            elif listMethods[indexMethod] == 'ODE':
                iterative = False
                print("iterative: ", iterative)
                MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial,
                                                                                 K_max, w, N, deltaT, MVirtual, MReal,
                                                                                 TVirtual, TReal, aLowerUpdated,
                                                                                 aUpperUpdated, it_idx,
                                                                                 update_cnt, iterative=iterative)

            else:
                if listMethods[indexMethod] == 'iMP':
                    iterative = True
                else:
                    iterative = False
                print("iterative: ", iterative)
                # use a child process to avoid CUDA context conflict
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

            outMOCUFile = open('../results/' + listMethods[indexMethod] + '_MOCU.txt', 'a')
            outTimeFile = open('../results/' + listMethods[indexMethod] + '_timeComplexity.txt', 'a')
            outSequenceFile = open('../results/' + listMethods[indexMethod] + '_sequence.txt', 'a')
            np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter="\t")
            np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter="\t")
            np.savetxt(outSequenceFile, experimentSequence, delimiter="\t")
            outMOCUFile.close()
            outTimeFile.close()
            outSequenceFile.close()
            save_MOCU_matrix[:, indexMethod, numberOfVaildSimulations] = MOCUCurve
        numberOfVaildSimulations += 1
    mean_MOCU_matrix = np.mean(save_MOCU_matrix, axis=2)
    print(mean_MOCU_matrix)
    outMOCUFile = open('../results/' + 'mean_MOCU.txt', 'a')
    np.savetxt(outMOCUFile, mean_MOCU_matrix, delimiter="\t")