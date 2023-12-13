#!/usr/bin/env python
# coding: utf-8
"""
Unit test for optDatasetConstrs
"""

import sys
import os
# add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from src.dataset import optDatasetConstrs
from src.model import tspDFJModel

import numpy as np

class testDataset(unittest.TestCase):
    def setUp(self):
        # set analog data for testing
        self.num_nodes = 10
        self.num_data = 5
        self.num_features = 10
        self.len_cost = self.num_nodes * (self.num_nodes - 1) // 2
        self.mock_feats = np.random.rand(self.num_data, self.num_features)
        self.mock_costs = np.random.rand(self.num_data, self.len_cost)

    def testDatasetCreation(self):
        """
        A test to check the soundness of optDatasetConstrs.
        """
        # optmodel
        optmodel = tspDFJModel(self.num_nodes)
        # creat dataset
        dataset = optDatasetConstrs(optmodel, self.mock_feats, self.mock_costs)
        # check datasize
        self.assertEqual(len(dataset), self.num_data,
                         "Dataset size mismatch: Expected {}, \
                         but got {}.".format(self.num_data, len(dataset)))
        # check optimal solution
        self.assertTrue(self.checkOptimal(dataset, optmodel),
                        "Solutions are not optimal.")
        # check binding constraints
        self.assertTrue(self.checkBinding(dataset, optmodel),
                        "Constraints are not binding to optimal")

    def checkOptimal(self, dataset, optmodel):
        """
        A method to check optimality
        """
        for i in range(len(dataset)):
            # get data per instance
            cost = dataset.costs[i]
            sol = dataset.sols[i]
            # set obj
            optmodel.setObj(cost)
            # solve
            _, obj = optmodel.solve()
            if not np.isclose(obj, cost@sol):
                print(obj, cost@sol)
                return False
        return True

    def checkBinding(self, dataset, optmodel):
        """
        A method to check binding constraints
        """
        for i in range(len(dataset)):
            # get data per instance
            cost = dataset.costs[i]
            ctrs = dataset.ctrs[i]
            sol1 = dataset.sols[i]
            # set obj
            optmodel.setObj(-ctrs.mean(axis=0))
            # solve
            sol2, obj = optmodel.solve()
            if not np.isclose(cost@sol1, cost@sol2):
                return False
        return True

if __name__ == "__main__":
    unittest.main()
