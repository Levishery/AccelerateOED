from io import StringIO
import numpy as np
import re
import matplotlib.pyplot as plt
from ast import literal_eval

result = '''[[1.52481979 1.52391604 1.52199242 1.5240857  1.52483089]
 [1.49073303 1.48894475 1.48940733 1.52175715 1.5196967 ]
 [1.48791281 1.48626036 1.48727119 1.52004748 1.5188121 ]
 [1.48622419 1.484633   1.48586169 1.51886485 1.51856894]
 [1.48609528 1.48381586 1.48425849 1.51754915 1.51807382]
 [1.4858397  1.48321005 1.48299165 1.5169798  1.5175849 ]
 [1.48576632 1.48245335 1.48299165 1.5152168  1.51749947]
 [1.48505299 1.48159031 1.482667   1.49385641 1.51700929]
 [1.48402753 1.48159031 1.48233951 1.49326457 1.51700929]
 [1.48312211 1.48107516 1.48220349 1.49318187 1.48677978]
 [1.48299502 1.48075807 1.48213027 1.49306894 1.48464049]]'''
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

result = '''[[0.30910667 0.30893524 0.30943992 0.30925237 0.30797995]
 [0.28789138 0.28858378 0.2880338  0.30831053 0.3071667 ]
 [0.28690723 0.28676778 0.28704665 0.30646866 0.30691106]
 [0.28610281 0.28616909 0.28684055 0.3060953  0.30689987]
 [0.28589432 0.28593781 0.2861669  0.29938677 0.30662962]
 [0.28540507 0.28567466 0.28601374 0.29892662 0.30607557]
 [0.28529727 0.28557092 0.28592827 0.29478297 0.28573236]
 [0.2851392  0.28548493 0.28570239 0.29445948 0.28542075]
 [0.28499207 0.28541688 0.28520292 0.29386407 0.28528914]
 [0.28498765 0.28529209 0.28515749 0.29372238 0.28521066]
 [0.2849702  0.2852557  0.28511955 0.28562268 0.28511464]]'''
result = re.sub(r"([^[])\s+([^]])", r"\1, \2", result)
result = np.array(literal_eval(result))

ax1 = axs.flat[1]
ax1.plot(x, result[:, 0], label='iMP')
ax1.plot(x, result[:, 1], label='MP')
ax1.plot(x, result[:, 2], label='OED')
ax1.plot(x, result[:, 3], label='RANDOM')
ax1.plot(x, result[:, 4], label='Entropy')
ax1.set_xlabel('Number of updates')
ax1.set_ylim([0.284, 0.312])
ax1.set_title("Experimental design for N=5 oscillators")
ax1.grid(True)
ax1.legend(loc = 'upper right')
plt.show()