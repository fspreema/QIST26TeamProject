"""Run from Fabian/Ising: python -m unittest local_solvers.test_adapters"""
import math
import unittest
from decimal import Decimal
from unittest.mock import patch

from wcnf_matrix import BoolVar, CNF, WeightFunction
from .adapters import (CachetExperimental, TensorOrderLocal, format_cachet_decimal,
                       format_mcc, parse_result)


class AdapterTests(unittest.TestCase):
    def test_mcc_preserves_both_weights_and_input(self):
        a, b = BoolVar(), BoolVar()
        cnf = CNF([[a, -b]])
        wf = WeightFunction([a, b])
        wf[a, True], wf[a, False] = 2., 3.
        wf[b, True], wf[b, False] = .25, .75
        before = wf.copy()
        text = format_mcc(cnf, wf)
        self.assertIn('w 1 2.0\nw -1 3.0', text)
        self.assertIn('w 2 0.25\nw -2 0.75', text)
        self.assertIn('1 -2 0', text)
        self.assertEqual(wf, before)

    def test_decimal_factor_and_raw_subnormal_rescaling(self):
        variables = [BoolVar() for _ in range(1400)]
        wf = WeightFunction(variables)
        wf.fill(1.)
        _, factor = format_cachet_decimal(CNF(), wf)
        self.assertTrue(factor.is_finite())
        self.assertGreater(factor, Decimal('1e400'))
        result = parse_result('Satisfying probability 2e-400', 'cachet', .2, Decimal('1e419'))
        self.assertEqual(result.model_count, 2e19)
        self.assertEqual(wf[variables[0], True], 1.)

    def test_tensor_output_uses_direct_count_and_solver_time(self):
        result = parse_result('Total Time: 0.7\nCount: 3.30493125e19\n', 'tensororder', 1.5)
        self.assertEqual(result.model_count, 3.30493125e19)
        self.assertEqual(result.runtime, .7)

    def test_cachet_weight_tokens_fit_legacy_parser(self):
        v = BoolVar()
        wf = WeightFunction([v])
        wf[v, True], wf[v, False] = 1., 2.
        text, factor = format_cachet_decimal(CNF(), wf)
        token = text.splitlines()[1].split()[2]
        self.assertLess(len(token), 64)
        self.assertAlmostEqual(float(token), 1/3)
        self.assertEqual(factor, Decimal(3))

    def test_invalid_output_is_not_success(self):
        for output in ('Count: inf\nTotal Time: 1', 'Count: nan\nTotal Time: 1',
                       'Count: 1\nTotal Time: -1', 'Count: 1'):
            with self.assertRaises(ValueError):
                parse_result(output, 'tensororder', 1.)

    def test_solver_failure_is_reported(self):
        with patch('local_solvers.adapters.subprocess.Popen', side_effect=OSError('Docker unavailable')):
            self.assertFalse(TensorOrderLocal().model_count(CNF(), WeightFunction([])).success)

    def test_timeouts_are_bounded(self):
        self.assertEqual(CachetExperimental()._timeout, 30)
        for limit in (0, -1, math.inf):
            with self.assertRaises(ValueError):
                TensorOrderLocal(timeout=limit)


if __name__ == '__main__':
    unittest.main()
