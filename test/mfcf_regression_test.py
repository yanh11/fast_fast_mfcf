import os
import unittest

import numpy as np
from tqdm import tqdm

print(os.getcwd())

import legacy_code.fast_mfcf_fixed4 as legacy_fast_mfcf

from fast_fast_mfcf import MFCF
from fast_fast_mfcf import mfcf_control


class TestMFCF(unittest.TestCase):
    def setUp(self):
        self.number_of_test_iterations = 100

    def _generate_symmetric_matrix(self, size):
        random_matrix = np.random.rand(size, size)
        return (random_matrix + random_matrix.T) / 2

    def _get_result(self, matrix_min_size, matrix_max_size, ctl):
        W = self._generate_symmetric_matrix(
            np.random.randint(matrix_min_size, matrix_max_size))

        gain_function = legacy_fast_mfcf.sumsquares_gen
        legacy_results = legacy_fast_mfcf.fast_mfcf(W, ctl, gain_function)

        new_results = MFCF().fast_mfcf(W, ctl, "sumsquares")

        return legacy_results, new_results

    def _assert_same_results(self, legacy_results, new_results):
        self.assertEqual(legacy_results[0], new_results[0])  # cliques
        self.assertEqual(legacy_results[1], new_results[1])  # separators
        self.assertEqual(legacy_results[2], new_results[2])  # peo
        # note that the new version does not use and return the gain table, so
        # we skip that comparison and use index 3 for J_logo
        np.testing.assert_allclose(legacy_results[4], new_results[3], rtol=1e-7,
                                   atol=1e-8, err_msg="J_logo matrices differ")

    def test_tmfg(self):
        print("Testing 'tmfg' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.00
        ctl['drop_sep'] = True
        ctl['min_clique_size'] = 4
        ctl['max_clique_size'] = 4
        ctl['method'] = 'TMFG'

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)

    def test_mst(self):
        print("Testing 'mst' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.00
        ctl['drop_sep'] = False
        ctl['min_clique_size'] = 1
        ctl['max_clique_size'] = 2
        ctl['coordination_number'] = np.inf

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)

    def test_mfcf(self):
        print("Testing 'mfcf' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.00
        ctl['drop_sep'] = False
        ctl['min_clique_size'] = 1
        ctl['max_clique_size'] = 10
        ctl['coordination_number'] = np.inf
        ctl['method'] = 'MFCF'

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)

    def test_threshold(self):
        print("Testing 'threshold' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.5
        ctl['drop_sep'] = False
        ctl['min_clique_size'] = 1
        ctl['max_clique_size'] = 10
        ctl['coordination_number'] = np.inf
        ctl['method'] = 'MFCF'

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)

    def test_sep_multiplicity(self):
        print("Testing 'sep_multiplicity' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.00
        ctl['drop_sep'] = False
        ctl['min_clique_size'] = 1
        ctl['max_clique_size'] = 10
        ctl['max_separator_multiplicity'] = 2
        ctl['method'] = 'MFCF'

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)

    def test_drop_sep(self):
        print("Testing 'drop_sep' setup...")
        ctl = mfcf_control()
        ctl['threshold'] = 0.00
        ctl['drop_sep'] = True
        ctl['min_clique_size'] = 1
        ctl['max_clique_size'] = 10
        ctl['coordination_number'] = np.inf
        ctl['method'] = 'MFCF'

        for _ in tqdm(range(self.number_of_test_iterations)):
            legacy_results, new_results = self._get_result(10, 50, ctl)
            self._assert_same_results(legacy_results, new_results)


if __name__ == "__main__":
    unittest.main()
