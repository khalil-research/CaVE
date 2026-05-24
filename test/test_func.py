#!/usr/bin/env python
"""
Unit test for CaVE function
"""

import os
import sys
import unittest
from unittest.mock import Mock

import torch

# add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyepo import EPO
from pyepo.model.opt import optModel
from src.cave import exactConeAlignedCosine, innerConeAlignedCosine


def _mock_optmodel(sense=EPO.MINIMIZE):
    """
    A mock optmodel that satisfies isinstance check and exposes modelSense
    """
    m = Mock(spec=optModel)
    m.modelSense = sense
    # bypass isinstance(opt, optModel) check in optModule.__init__
    m.__class__ = optModel
    return m


class TestExactConeAlignedCosine(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        # a mock optmodel with the proper EPO enum for modelSense
        self.mock_optmodel = _mock_optmodel(EPO.MINIMIZE)
        # set analog data for testing
        self.len_cost = 10
        self.num_binding_constrs = 15
        self.batch_size = 32
        self.mock_costs = torch.rand(self.batch_size, self.len_cost)
        self.mock_bctrs = torch.rand(self.batch_size, self.num_binding_constrs, self.len_cost)

    def testNNLS(self):
        """
        A test to check the loss forward pass goes well with scipy.nnls.
        """
        # init loss
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="nnls")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())

    def testClarabel(self):
        """
        A test to check the loss forward pass goes well with Clarabel.
        """
        # init loss
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="clarabel")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())


class TestInnerConeAlignedCosine(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        # a mock optmodel with the proper EPO enum for modelSense
        self.mock_optmodel = _mock_optmodel(EPO.MINIMIZE)
        # set analog data for testing
        self.len_cost = 10
        self.num_binding_constrs = 15
        self.batch_size = 32
        self.mock_costs = torch.rand(self.batch_size, self.len_cost)
        self.mock_bctrs = torch.rand(self.batch_size, self.num_binding_constrs, self.len_cost)

    def testNNLS(self):
        """
        A test to check the loss forward pass goes well with scipy.nnls.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", seed=42)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())

    def testNNLSPartial(self):
        """
        A test to check the loss forward pass goes well with the heuristic projection.
        """
        # init loss with solve_ratio=0 (heuristic-only branch)
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0, seed=42)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())

    def testClarabel(self):
        """
        A test to check the loss forward pass goes well with Clarabel.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel", seed=42)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())

    def testClarabelPartial(self):
        """
        A test to check the loss forward pass goes well with the heuristic projection.
        """
        # init loss with solve_ratio=0 (heuristic-only branch)
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel", solve_ratio=0, seed=42)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())


class TestRegressionFixes(unittest.TestCase):
    """
    Regression tests for bugs fixed in the optModule-based rewrite.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.mock_optmodel = _mock_optmodel(EPO.MINIMIZE)

    def testZeroPaddedRowsMaskedInHeuristic(self):
        """
        Padded zero rows must not pull the heuristic average toward zero.
        """
        # one batch with no padding, one with extra zero rows
        pred = torch.rand(2, 6)
        ctrs_full = torch.rand(2, 5, 6)
        ctrs_padded = torch.cat([ctrs_full, torch.zeros(2, 10, 6)], dim=1)
        # init loss in heuristic-only branch
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0, seed=42)
        # forward pass both
        loss_full = cave(pred, ctrs_full)
        loss_padded = cave(pred, ctrs_padded)
        # results should match: padded rows are masked out of the average
        self.assertTrue(torch.isclose(loss_full, loss_padded, atol=1e-6).item(),
                        "Padded rows changed the heuristic loss.")

    def testProjNormDivisionByZeroSafe(self):
        """
        Pure-zero prediction must not produce NaN via proj/proj.norm().
        """
        # zero prediction triggers a degenerate projection direction
        pred = torch.zeros(2, 6)
        ctrs = torch.rand(2, 3, 6)
        # forward pass
        loss = exactConeAlignedCosine(self.mock_optmodel, solver="nnls")(pred, ctrs)
        # check loss is finite (no NaN/Inf)
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is NaN/Inf for zero prediction.")

    def testInvalidSolverRaises(self):
        """
        Unknown solver name must raise ValueError at construction.
        """
        with self.assertRaises(ValueError):
            exactConeAlignedCosine(self.mock_optmodel, solver="bogus")

    def testInvalidSolveRatioRaises(self):
        """
        solve_ratio outside [0, 1] must raise ValueError at construction.
        """
        with self.assertRaises(ValueError):
            innerConeAlignedCosine(self.mock_optmodel, solve_ratio=1.5)

    def testInvalidInnerRatioRaises(self):
        """
        inner_ratio outside [0, 1] must raise ValueError at construction.
        """
        with self.assertRaises(ValueError):
            innerConeAlignedCosine(self.mock_optmodel, inner_ratio=-0.1)

    def testSeedDeterminism(self):
        """
        Same seed must give the same hybrid branch decision and loss.
        """
        pred = torch.rand(4, 6)
        ctrs = torch.rand(4, 5, 6)
        # init two losses with identical seed
        l1 = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0.5, seed=7)(pred, ctrs)
        l2 = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0.5, seed=7)(pred, ctrs)
        # results should match
        self.assertTrue(torch.isclose(l1, l2).item(),
                        "Same seed produced different losses.")


class TestAPGD(unittest.TestCase):
    """
    APGD solver tests. Requires torch.compile backend (Linux/WSL/GPU).
    """

    def setUp(self):
        torch.manual_seed(0)
        self.mock_optmodel = _mock_optmodel(EPO.MINIMIZE)
        # set analog data for testing
        self.len_cost = 10
        self.num_binding_constrs = 15
        self.batch_size = 8
        self.mock_costs = torch.rand(self.batch_size, self.len_cost)
        self.mock_bctrs = torch.rand(self.batch_size, self.num_binding_constrs, self.len_cost)

    def testExactAPGD(self):
        """
        A test to check the exact APGD forward pass goes well.
        """
        # init loss with explicit APGD
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="apgd")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")
        # cuda exist or not
        if torch.cuda.is_available():
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())

    def testInnerAPGD(self):
        """
        A test to check the inner APGD forward pass and auto-injected truncation.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="apgd", seed=42)
        # check truncation kwargs got auto-injected
        self.assertEqual(cave.solver_kwargs, {"max_iters": 20, "tol_grad": None},
                         "inner APGD did not auto-inject truncation kwargs.")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # check loss is finite
        self.assertTrue(torch.isfinite(loss).item(),
                        "Loss is not finite.")

    def testAPGDvsClarabelClose(self):
        """
        A test to check APGD projection direction is close to Clarabel converged.
        """
        import torch.nn.functional as F
        # non-degenerate cone: use Gaussian (both signs) so projection is non-trivial
        torch.manual_seed(1)
        costs = torch.randn(self.batch_size, self.len_cost)
        bctrs = torch.randn(self.batch_size, self.num_binding_constrs, self.len_cost)
        # apgd projection (exact mode)
        cave_apgd = exactConeAlignedCosine(self.mock_optmodel, solver="apgd")
        with torch.no_grad():
            signed = -1.0 * costs
            proj_apgd = cave_apgd._get_projection(signed, bctrs)
        # clarabel projection (exact mode)
        cave_cb = exactConeAlignedCosine(self.mock_optmodel, solver="clarabel")
        with torch.no_grad():
            proj_cb = cave_cb._get_projection(signed, bctrs)
        # cosine similarity should be near 1
        cos = F.cosine_similarity(proj_apgd, proj_cb, dim=1)
        self.assertGreater(cos.min().item(), 0.95,
                           "APGD projection direction diverged from Clarabel.")

    def testAPGDInnerOverride(self):
        """
        A test to check user-supplied solver_kwargs override inner-mode defaults.
        """
        # user-supplied kwargs should NOT be replaced by inner defaults
        custom = {"max_iters": 50, "tol_grad": 1e-3}
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="apgd",
                                       solver_kwargs=custom, seed=42)
        self.assertEqual(cave.solver_kwargs, custom,
                         "User solver_kwargs got overridden by inner defaults.")


if __name__ == "__main__":
    unittest.main()
