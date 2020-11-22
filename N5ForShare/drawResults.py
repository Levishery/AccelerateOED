import sys
import time

sys.path.append("./src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

update_cnt = 10

listMethods = ['iNN', 'NN', 'iODE', 'ODE', 'ENTROPY', 'RANDOM']
resultFolder = './resultsOnLambda_100/'

drawIndex = 0
iNN = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
iNN = iNN.mean(0)
iNNT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
iNNT = iNNT.mean(0)

drawIndex = 1
NN = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
NN = NN.mean(0)
NNT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
NNT = NNT.mean(0)

# drawIndex = 2
# iODE = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
# iODE = iODE.mean(0)
# iODET = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
# iODET = iODET.mean(0)

drawIndex = 3
ODE = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
ODE = ODE.mean(0)
ODET = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
ODET = ODET.mean(0)

drawIndex = 4
ENT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
ENT = ENT.mean(0)
ENTT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
ENTT = ENTT.mean(0)

drawIndex = 5
RND = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
RND = RND.mean(0)
RNDT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
RNDT = RNDT.mean(0)

x_ax = np.arange(0,update_cnt+1,1)
# plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
# plt.legend(['NN-based MOCU (iterative)', 'NN-based MOCU', 'Entropy-based', 'Random'])
plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, ODE, 'yo--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
plt.legend(['Proposed (iterative)', 'Proposed', 'ODE', 'Entropy', 'Random'])
# plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, iODE, 'yp:', x_ax, ODE, 'yo--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
# plt.legend(['NN-based MOCU (iterative)', 'NN-based MOCU', 'ODE-based MOCU (iterative)', 'ODE-based MOCU', 'Entropy-based', 'Random'])
plt.xticks(np.arange(0, update_cnt+1,1)) 
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
# plt.title('Experimental design for N=5 oscillators')
plt.grid(True)
plt.savefig(resultFolder + "MOCU_5.eps")

x_ax = np.arange(0, update_cnt+1, 1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(x_ax, np.insert(np.cumsum(iNNT), 0, 0.0000000001), 'r*:', x_ax, np.insert(np.cumsum(NNT), 0, 0.0000000001), 'rs--', x_ax, np.insert(np.cumsum(ODET), 0, 0.0000000001), 'yo--')
plt.legend(['Proposed (iterative)', 'Proposed', 'ODE'])
# plt.plot(x_ax, np.insert(np.cumsum(iNNT), 0, 0.0000000001), 'r*:', x_ax, np.insert(np.cumsum(NNT), 0, 0.0000000001), 'rs--', x_ax, np.insert(np.cumsum(ODET), 0, 0.0000000001), 'yo--', x_ax, np.insert(np.cumsum(ENTT), 0, 0.0000000001), 'gd:', x_ax, np.insert(np.cumsum(RNDT), 0, 0.0000000001), 'b,:')
# plt.legend(['Proposed (iterative)', 'Proposed', 'ODE', 'Entropy', 'Random'])
# plt.plot(x_ax, iNNT, 'r*:', x_ax, NNT, 'rs--', x_ax, iODET, 'yp:', x_ax, ODET, 'yo--', x_ax, ENTT, 'gd:', x_ax, RNDT, 'b,:')
# plt.legend(['NN-based MOCU (iterative)', 'NN-based MOCU', 'ODE-based MOCU (iterative)', 'ODE-based MOCU', 'Entropy-based', 'Random'])
plt.yscale('log')
plt.xlabel('Number of updates')
plt.ylabel('Cumulative time complexity (in seconds)')
plt.xticks(np.arange(0, update_cnt+1,1)) 
# plt.ylim(0.000001, 10000)
plt.ylim(1, 10000)
# plt.title('Time complexityExperimental design for N=5 oscillators')
plt.grid(True)
fig.savefig(resultFolder + 'timeComplexity_5.eps')
plt.close(fig)

