from fast_fast_mfcf import MFCF
from fast_fast_mfcf import mfcf_control
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)


np.random.seed(42)
size = 100
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

cliques, separators, peo, J_logo = MFCF().fast_mfcf(C, ctl, "sumsquares")
print(cliques)
print(separators)
