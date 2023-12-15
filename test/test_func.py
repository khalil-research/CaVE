#!/usr/bin/env python
# coding: utf-8
"""
Unit test for CaVE function
"""

import sys
import os
# add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from unittest.mock import Mock
from src.cave import exactConeAlignedCosine, innerConeAlignedCosine

import torch
from pyepo.model.opt import optModel

class TestExactConeAlignedCosine(unittest.TestCase):

    def setUp(self):
        # a mock optmodel with only modelSense
        self.mock_optmodel = Mock(optModel)
        self.mock_optmodel.modelSense = 'MINIMIZE'
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
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="nnls", processes=4)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")

    def testClarabel(self):
        """
        A test to check the loss forward pass goes well with Clarabel.
        """
        # init loss
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="clarabel")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = exactConeAlignedCosine(self.mock_optmodel, solver="clarabel", processes=4)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")


class TestInnerConeAlignedCosine(unittest.TestCase):

    def setUp(self):
        # a mock optmodel with only modelSense
        self.mock_optmodel = Mock(optModel)
        self.mock_optmodel.modelSense = 'MINIMIZE'
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
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", processes=4)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")


    def testNNLSPartial(self):
        """
        A test to check the loss forward pass goes well with heuristic proejection.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="nnls", solve_ratio=0, processes=4)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")

    def testClarabel(self):
        """
        A test to check the loss forward pass goes well with Clarabel.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel")
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel", processes=4)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")

    def testClarabelPartial(self):
        """
        A test to check the loss forward pass goes well with heuristic proejection.
        """
        # init loss
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel", solve_ratio=0)
        # forward pass
        loss = cave(self.mock_costs, self.mock_bctrs)
        # cuda exist or not
        if torch.cuda.is_available():
            # forward pass with cuda
            cave(self.mock_costs.cuda(), self.mock_bctrs.cuda())
        # multiprocessing
        cave = innerConeAlignedCosine(self.mock_optmodel, solver="clarabel", processes=4, solve_ratio=0)
        # forward pass
        loss_mp = cave(self.mock_costs, self.mock_bctrs)
        # check result consistency
        self.assertTrue(torch.isclose(loss, loss_mp).item(),
                        "Inconsistent results with multiple processes.")


if __name__ == "__main__":
    unittest.main()
