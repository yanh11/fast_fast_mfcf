# Fast Fast MFCF

**Fast Fast MFCF** is an optimised implementation of the *Maximally Filtered Clique Forest (MFCF)* algorithm.  
The original algorithm was proposed by Guido Previde Massara, and this Python implementation is based on the 
[original version](https://github.com/FinancialComputingUCL/MFCF) by Hongyu (Eric) Lin and Tomaso Aste.

---

## 🚀 Performance

Our implementation significantly accelerates MFCF generation:

- **~100× faster** for networks with 200 vertices, and even faster for larger networks
- The time required to generate a TMFG with **200 vertices** can now generate one with **5,000 vertices**
- Check the full performance benchmark [here](test/plot.ipynb)
---

## 🧩 Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/yanh11/fast_fast_mfcf.git
cd fast_fast_mfcf
pip install -r requirements.txt
```

---

## 🧠 Usage
You can create an MFCF with customised control from a correlation matrix:

```python
import numpy as np
from fast_fast_mfcf import MFCF, mfcf_control

ctl = mfcf_control()
ctl['threshold'] = 0.00
ctl['drop_sep'] = False
ctl['min_clique_size'] = 4
ctl['max_clique_size'] = 15
ctl['coordination_number'] = np.inf
ctl['method'] = 'MFCF'

cliques, separators, peo, J_log = MFCF().fast_mfcf(C, ctl, gf_type="sumsquares")
```

Outputs:
* `cliques`: list of maximal cliques
* `separators`: list of separators
* `peo`: perfect elimination ordering
* `J_log`: logo matrix
