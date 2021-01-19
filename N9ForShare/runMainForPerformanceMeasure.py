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
    it_idx = 10
    update_cnt = 18
    N = 9
    K_max = 20480

    # Time must be larger than 250 for N = 5
    deltaT = 1.0/160.0
    TVirtual = 5
    MVirtual = int(TVirtual/deltaT)
    TReal = 5
    MReal = int(TReal/deltaT)

    w = np.zeros(N)
    w[0] = 1.19
    w[1] = 3.23
    w[2] = 6.34
    w[3] = 7.48
    w[4] = 10.9
    w[5] = 11.62
    w[6] = 14.74
    w[7] = 29.58
    w[8] = 38.88

    listMethods = ['iMP', 'MP', 'ODE', 'RANDOM', 'ENTROPY']
    numberOfSimulationsPerMethod = 1
    numberOfVaildSimulations = 0
    numberOfSimulations = 0

    aInitialUpper = np.zeros((N,N))
    aInitialLower = np.zeros((N,N))

    for i in range(N):
        for j in range(i+1,N):
            if (i <= 4) and (j <= 4):
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.03
                aInitialLower[i, j] = f_inv*0.97
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]
            elif (i >= 7) or (j >= 7):
                f_inv = np.abs(w[i] - w[j])/9.0
                aInitialUpper[i, j] = f_inv*1.02
                aInitialLower[i, j] = f_inv*0.98
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]
            else:
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.01
                aInitialLower[i, j] = f_inv*0.99
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]

            if j == 5:
                f_inv = np.abs(w[i] - w[j])/2.0
                aInitialUpper[i, j] = f_inv*1.02
                aInitialLower[i, j] = f_inv*0.98
                aInitialUpper[j, i] = aInitialUpper[i, j]
                aInitialLower[j, i] = aInitialLower[i, j]

    np.savetxt('./results/paramNaturalFrequencies.txt', w, fmt='%.64e')
    np.savetxt('./results/paramInitialUpper.txt', aInitialUpper, fmt='%.64e')
    np.savetxt('./results/paramInitialLower.txt', aInitialLower, fmt='%.64e')

    while (numberOfSimulationsPerMethod > numberOfVaildSimulations):
        randomState = np.random.RandomState(int(numberOfSimulations))
        a = np.zeros((N,N))
        # for i in range(N):
        #     for j in range(i+1,N):
        #         randomNumber = randomState.uniform()
        #         a[i,j] = aInitialLower[i,j] + randomNumber*(aInitialUpper[i,j] - aInitialLower[i,j])
        #         a[j,i] = a[i,j]

        r1 = 3.0
        r2 = 3.0
        r3 = 3.0
        for i in range(N):
            for j in range(N):
                if i==2:
                    a[i,j] = aInitialLower[i,j] + 0.5*(aInitialUpper[i,j] - aInitialLower[i,j])
                elif i==4:
                    a[i,j] = aInitialLower[i, j] + r2/6.0*(aInitialUpper[i, j] - aInitialLower[i,j])
                elif i==6:
                    a[i,j] = aInitialLower[i, j] + r3/6.0 *(aInitialUpper[i,j] - aInitialLower[i,j])
                else:
                    a[i,j] = aInitialLower[i, j] + r1/6.0*(aInitialUpper[i,j] - aInitialLower[i,j])
                a[j,i] = a[i,j]


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

        np.savetxt('./results/paramCouplingStrength' + str(numberOfVaildSimulations) + '.txt', a, fmt='%.64e')
        test = np.loadtxt('./results/paramCouplingStrength' + str(numberOfVaildSimulations) + '.txt')
        for indexMethod in range(len(listMethods)):
            timeMOCU = time.time()
            MOCUInitial = MOCU(K_max, w, N, deltaT, MReal, TReal, aInitialLower.copy(), aInitialUpper.copy(), 0)
            print("Round: ", numberOfVaildSimulations, "/", numberOfSimulationsPerMethod, "-", listMethods[indexMethod], "Iteration: ", numberOfVaildSimulations, " Initial MOCU: ", MOCUInitial, " Computation time: ", time.time() - timeMOCU)
            aUpperUpdated = aInitialUpper.copy()
            aLowerUpdated = aInitialLower.copy()
            if listMethods[indexMethod] == 'RANDOM':
                MOCUCurve, experimentSequence, timeComplexity = findRandomSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MReal, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt)

            elif listMethods[indexMethod] == 'ENTROPY':
                MOCUCurve, experimentSequence, timeComplexity = findEntropySequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MReal, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt)

            elif listMethods[indexMethod] == 'iODE':
                iterative = True
                print("iterative: ", iterative)
                MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial,
                        K_max, w, N, deltaT, MVirtual, MReal, TVirtual, TReal, aLowerUpdated, aUpperUpdated, it_idx,
                        update_cnt, iterative=iterative)

            elif listMethods[indexMethod] == 'ODE':
                iterative = False
                print("iterative: ", iterative)
                MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MVirtual, MReal, TVirtual, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt, iterative = iterative)

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
                                                                              w, N, aLowerUpdated, aUpperUpdated, update_cnt,
                                                                              iterative))
                context.join()
                [optimalExperiments, update_bond, timeComplexity] = q.get()

                MOCUCurve, experimentSequence, timeComplexity = findMPSequence(optimalExperiments, update_bond,
                                                                               timeComplexity, MOCUInitial, K_max, w, N, deltaT,
                                                                               MVirtual, MReal, TVirtual, TReal,
                                                                               aLowerUpdated, aUpperUpdated, it_idx, update_cnt)

            outMOCUFile = open('./results/' + listMethods[indexMethod] + '_MOCU.txt', 'a')
            outTimeFile = open('./results/' + listMethods[indexMethod] + '_timeComplexity.txt', 'a')
            outSequenceFile = open('./results/' + listMethods[indexMethod] + '_sequence.txt', 'a')
            np.savetxt(outMOCUFile, MOCUCurve.reshape(1, MOCUCurve.shape[0]), delimiter = "\t")
            np.savetxt(outTimeFile, timeComplexity.reshape(1, timeComplexity.shape[0]), delimiter = "\t")
            np.savetxt(outSequenceFile, experimentSequence, delimiter = "\t")
            outMOCUFile.close()
            outTimeFile.close()
            outSequenceFile.close()
        numberOfVaildSimulations += 1