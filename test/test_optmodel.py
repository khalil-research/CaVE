#!/usr/bin/env python
# coding: utf-8
"""
Unit test for optmodel
"""

import sys
import os
# add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from collections import defaultdict
from src.model import tspDFJModel, vrpModel

import numpy as np

class testTSPSolver(unittest.TestCase):
    def setUp(self):
        # number of nodes
        self.num_nodes = 20
        # edges of full connected graph
        self.edges = [(u, v) for u in range(self.num_nodes)
                      for v in range(self.num_nodes) if u < v]
        # random cost
        self.cost = np.random.rand(self.num_nodes*(self.num_nodes-1)//2)

    def testSolution(self):
        """
        A test to check the feasibility of solution.
        """
        # optmodel
        optmodel = tspDFJModel(self.num_nodes)
        # set obj
        optmodel.setObj(self.cost)
        # solve
        sol, obj = optmodel.solve()
        # sol to edges
        active_edges = self.getActiveEdge(sol)
        # check all visited
        self.assertTrue(self.checkVisited(active_edges),
                        "The solution should visit each city exactly once.")
        # check no subtour
        self.assertTrue(self.checkSubtour(active_edges),
                        "The solution has a subtour.")

    def getActiveEdge(self, sol):
        """
        A method to get active edges
        """
        active_edges = []
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > 1e-2:
                active_edges.append((u, v))
        return active_edges

    def checkVisited(self, active_edges):
        """
        A method to check all nodes are visted once
        """
        # count visited
        node_counts = defaultdict(int)
        for i, j in active_edges:
            node_counts[i] += 1
            node_counts[j] += 1
        # check if each node connects exactly two edges
        if any(count != 2 for count in node_counts.values()):
            return False
        else:
            return True

    def checkSubtour(self, active_edges):
        """
        A method to check if exist a subtour
        """
        # init visited nodes
        visited = set()
        # start
        current = active_edges[0][0]
        while True:
            visited.add(current)
            # find next node
            next_node = None
            for i, j in self.edges:
                if i == current and j not in visited:
                    next_node = j
                    break
                elif j == current and i not in visited:
                    next_node = i
                    break
            # no node in tour is unvisited
            if next_node is None:
                break
            current = next_node
        # visited all nodes  in a tour
        return len(visited) == self.num_nodes


class testCVRPSolver(unittest.TestCase):
    def setUp(self):
        # number of nodes
        self.num_nodes = 20
        # vehicle capacity
        self.cap = 30
        # number of vehicles
        self.num_vehicles = 5
        # edges of full connected graph
        self.edges = [(u, v) for u in range(self.num_nodes+1)
                      for v in range(self.num_nodes+1) if u < v]
        # demands
        self.demands = np.random.rand(self.num_nodes) * 10
        # random cost
        self.cost = np.random.rand(self.num_nodes*(self.num_nodes+1)//2)

    def testSolution(self):
        """
        A test to check the feasibility of solution.
        """
        # optmodel
        optmodel = vrpModel(self.num_nodes+1, self.demands, self.cap, self.num_vehicles)
        # set obj
        optmodel.setObj(self.cost)
        # solve
        sol, obj = optmodel.solve()
        # sol to edges
        active_edges = self.getActiveEdge(sol)
        # check all visited
        self.assertTrue(self.checkVisited(active_edges),
                        "The solution should visit each city exactly once.")
        # check the number of vehicles
        self.assertTrue(self.checkVehicleCount(active_edges),
                        "The number of vehicles used should not exceed the allowed limit.")
        # check validity tour
        self.assertTrue(self.checkSubtour(active_edges),
                        "The solution has a subtour.")
        # check if all demands are met and no vehicle exceeds its capacity
        self.assertTrue(self.checkCapacity(active_edges),
                        "All demands must be met without exceeding vehicle capacities.")

    def getActiveEdge(self, sol):
        """
        A method to get active edges
        """
        active_edges = []
        for i, (u, v) in enumerate(self.edges):
            if sol[i] > 1e-2:
                active_edges.append((u, v))
        return active_edges

    def checkVisited(self, active_edges):
        """
        A method to check all nodes are visted once
        """
        # count visited
        node_counts = defaultdict(int)
        # skip depot
        for i, j in active_edges:
            if i != 0:
                node_counts[i] += 1
            if j != 0:
                node_counts[j] += 1
        # check if each node connects exactly two edges
        if any(count != 2 for count in node_counts.values()):
            return False
        else:
            return True

    def checkVehicleCount(self, active_edges):
        """
        A method to check the number of vehicles that does not exceed the limit.
        """
        depot_starts = sum(1 for edge in active_edges if edge[0] == 0)
        return depot_starts <= 2 * self.num_vehicles

    def checkSubtour(self, active_edges):
        """
        A method to check if exist a subtour
        """
        # adjacency list
        node_edges = defaultdict(list)
        for i, j in active_edges:
            node_edges[i].append(j)
            node_edges[j].append(i)
        # init visited nodes
        visited = set()
        # starting from the depot
        to_visit = [0]
        while to_visit:
            # add to visited
            current = to_visit.pop()
            visited.add(current)
            # add neighbors
            for neighbor in node_edges[current]:
                if neighbor not in visited:
                    to_visit.append(neighbor)
       # check if all nodes are reachable
        return len(visited) == self.num_nodes + 1

    def checkCapacity(self, active_edges):
        """
        A method to check no exceeding of vehicle capacities.
        """
        # get list of tours
        tours = self.getTours(active_edges)
        # check each tour
        for tour in tours:
            load = 0
            # [1:-1] remove the depot on the tour
            for node in tour[1:-1]:
                load += self.demands[node-1]
                # exceeded capacity
                if load > self.cap:
                    print("The load of vehicle is {} that exceed the capacity {}".format(load, self.cap))
                    return False
        return True

    def getTours(self, active_edges):
        """
        A method to get a list of vehicle tours.
        """
        # adjacency list
        node_edges = defaultdict(list)
        for i, j in active_edges:
            node_edges[i].append(j)
            node_edges[j].append(i)
        # extract tours
        tours = []
        while node_edges[0]:  # While depot has edges
            tour = [0]
            current = 0
            while True:
                next_node = node_edges[current].pop(0)
                # also remove symetrically
                node_edges[next_node].remove(current)
                # add new node
                tour.append(next_node)
                # go back to depot
                if next_node == 0:
                    break
                current = next_node
            tours.append(tour)
        return tours


if __name__ == "__main__":
    unittest.main()
