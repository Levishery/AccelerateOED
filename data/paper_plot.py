from io import StringIO
import numpy as np
import re
import matplotlib.pyplot as plt
from ast import literal_eval

result = '''[[1.52547331 1.52495967 1.52579616 1.5240202  1.52556537]                                                                                                          
  [1.48903314 1.48938158 1.48856196 1.52127769 1.52170014]                                                                                                          
  [1.48766359 1.48730664 1.48660296 1.52091857 1.52070068]                                                                                                          
  [1.48658909 1.48553214 1.48473679 1.49809859 1.52021302]                                                                                                          
  [1.48552401 1.48495856 1.48473679 1.49581011 1.51950425]                                                                                                          
  [1.48338961 1.48484246 1.48402936 1.49493905 1.51883645]                                                                                                          
  [1.48338961 1.48483867 1.48333304 1.4934829  1.51833454]                                                                                                          
  [1.48273106 1.48417207 1.48319824 1.49345292 1.51826086]                                                                                                          
  [1.48239205 1.48304538 1.48309848 1.49345292 1.5176496 ]                                                                                                          
  [1.48140333 1.48298711 1.48309848 1.49339457 1.48793508]                                                                                                          
  [1.48140333 1.48265471 1.48147047 1.49267267 1.48580957]]'''
result = re.sub(r"([^[])\s+([^]])", r"\1, \2", result)
result = np.array(literal_eval(result))

plt.style.use('seaborn')
x = np.linspace(0, 10, 11)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
ax1 = axs.flat[0]
ax1.plot(x, result[:, 0], label='iMP')
ax1.plot(x, result[:, 1], label='MP')
ax1.plot(x, result[:, 2], label='OED')
ax1.plot(x, result[:, 3], label='RANDOM')
ax1.plot(x, result[:, 4], label='Entropy')
ax1.set_xlabel('Number of updates')
ax1.set_ylabel('MOCU')
ax1.set_ylim([1.478, 1.528])
ax1.set_title("Experimental design for N=7 oscillators")
ax1.grid(True)

result = '''[[0.30855266 0.30887809 0.3082809  0.30896352 0.30828622]                                                                                                          
  [0.28804178 0.28806198 0.28813954 0.30647203 0.30714565]                                                                                                          
  [0.28757247 0.28760276 0.28739186 0.30619909 0.30701084]                                                                                                          
  [0.28668311 0.2864712  0.28659434 0.30373369 0.30698222]                                                                                                          
  [0.28580588 0.28585933 0.28630934 0.30346125 0.30676094]                                                                                                          
  [0.28563297 0.28543565 0.28603408 0.30321088 0.30653502]                                                                                                          
  [0.28543634 0.2852517  0.28577607 0.30151391 0.28562637]                                                                                                          
  [0.28535586 0.28524175 0.28564505 0.29264736 0.28511685]                                                                                                          
  [0.28502354 0.28514505 0.28564505 0.29220306 0.28491946]                                                                                                          
  [0.28497113 0.2850693  0.28534398 0.28755444 0.28490278]                                                                                                          
  [0.28492623 0.28501705 0.2853193  0.28529369 0.28477725]]'''
result = re.sub(r"([^[])\s+([^]])", r"\1, \2", result)
result = np.array(literal_eval(result))

ax1 = axs.flat[1]
ax1.plot(x, result[:, 0], label='iMP')
ax1.plot(x, result[:, 1], label='MP')
ax1.plot(x, result[:, 2], label='Sampling-RK')
ax1.plot(x, result[:, 3], label='RANDOM')
ax1.plot(x, result[:, 4], label='Entropy')
ax1.set_xlabel('Number of updates')
ax1.set_ylim([0.284, 0.312])
ax1.set_title("Experimental design for N=5 oscillators")
ax1.grid(True)
ax1.legend(loc = 'upper right')
# plt.show()
plt.savefig('simulation.pdf')