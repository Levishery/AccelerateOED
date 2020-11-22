# Youngjoon Hong (07.15.2020)
# !!!!!!!!!!!!!!!!!!!!Important!!!!!!!!!!!!!!!!!!!!!!!
# If you want to chage the number of oscillators, Change "N" here.
# In addition, please do not forget update "manually" the followings in the MOCU.py file
# #define N_global 10. This is much faster than passing the params in pyCUDA.

import os
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\HostX64\x64"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

import sys
import time

sys.path.append("./src")

from findMOCUSequence import *
from findEntropySequence  import *
from findRandomSequence import *
from determineSyncTwo import *
from determineSyncN import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

it_idx = 10
update_cnt = 10
N = 5
K_max = 20480

# Time must be larger than 250 for N = 5
deltaT = 1.0/160.0
TVirtual = 5
MVirtual = int(TVirtual/deltaT)
TReal = 5
MReal = int(TReal/deltaT)

w = np.zeros(N)
w[0] = -2.5000
w[1] = -0.6667
w[2] = 1.1667
w[3] = 2.0000
w[4] = 5.8333

listMethods = ['RANDOM', 'ENTROPY', 'iODE', 'ODE']
numberOfSimulationsPerMethod = 1
numberOfVaildSimulations = 0
numberOfSimulations = 0

aInitialUpper = np.zeros((N,N))
aInitialLower = np.zeros((N,N))
for i in range(N):
    for j in range(i+1,N):
        syncThreshold = np.abs(w[i] - w[j])/2.0
        aInitialUpper[i, j] = syncThreshold * 1.15
        aInitialLower[i, j] = syncThreshold * 0.85
        aInitialUpper[j, i] = aInitialUpper[i, j]
        aInitialLower[j, i] = aInitialLower[i, j]
    
aInitialUpper[0,2:5] = aInitialUpper[0,2:5]*0.3
aInitialLower[0,2:5] = aInitialLower[0,2:5]*0.3
aInitialUpper[1,3:5] = aInitialUpper[1,3:5]*0.45
aInitialLower[1,3:5] = aInitialLower[1,3:5]*0.45

np.savetxt('./results/paramNaturalFrequencies.txt', w, fmt='%.64e')
np.savetxt('./results/paramInitialUpper.txt', aInitialUpper, fmt='%.64e')
np.savetxt('./results/paramInitialLower.txt', aInitialLower, fmt='%.64e')

for i in range(N):
    for j in range(i+1,N):
        aInitialUpper[j, i] = aInitialUpper[i, j]
        aInitialLower[j, i] = aInitialLower[i, j]

while (numberOfSimulationsPerMethod > numberOfVaildSimulations):
    randomState = np.random.RandomState(int(numberOfSimulations))
    a = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(i+1,N):
    #         randomNumber = randomState.uniform()
    #         a[i,j] = aInitialLower[i,j] + randomNumber*(aInitialUpper[i,j] - aInitialLower[i,j])
    #         a[j,i] = a[i,j]

    numberOfSimulations += 1

    r1 = 3.0
    r2 = 3.0
    r3 = 3.0

    for i in range(N):
        for j in range(i+1,N):
            if (i==1):
                a[i,j] = aInitialLower[i,j] + r1/6.0*(aInitialUpper[i,j] - aInitialLower[i,j])
                #a[j,i] = a[i,j]
            elif (i==2):
                a[i,j] = aInitialLower[i, j] + r2/6.0*(aInitialUpper[i,j] - aInitialLower[i,j])
                #a[j,i] = a[i,j]
            else:
                a[i,j] = aInitialLower[i, j] + r3/6.0*(aInitialUpper[i,j] - aInitialLower[i,j])
                #a[j,i] = a[i,j]
            a[j,i] = a[i,j]

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
        else:
            if listMethods[indexMethod] == 'iODE':
                iterative = True
            else:
                iterative = False
            print("iterative: ", iterative)
            MOCUCurve, experimentSequence, timeComplexity = findMOCUSequence(criticalK, isSynchronized, MOCUInitial, K_max, w, N, deltaT, MVirtual, MReal, TVirtual, TReal, aLowerUpdated, aUpperUpdated, it_idx, update_cnt, iterative = iterative)
        
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