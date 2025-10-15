from fast_mfcf_fixed4 import fast_mfcf
from fast_mfcf_fixed4 import sumsquares_gen
from fast_mfcf_fixed4 import mfcf_control
import numpy as np

np.random.seed(42)
size = 10
random_matrix = np.random.rand(size, size)
# Make it symmetric by averaging with its transpose
C = (random_matrix + random_matrix.T) / 2

ctl = mfcf_control()

ctl['threshold'] = 0.5
ctl['drop_sep'] = False
ctl['min_clique_size'] = 1
ctl['max_clique_size'] = 10
ctl['coordination_number'] = np.inf
ctl['method'] = 'MFCF'

cliques, separators, peo, gt, J_logo = fast_mfcf(C, ctl, sumsquares_gen)
print(cliques)
print(separators)
