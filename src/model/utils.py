#!/usr/bin/env python
# coding: utf-8
"""
Utilities
"""

from collections import defaultdict

class unionFind:
    """
    Union-find disjoint sets that provides methods to perform find and union
    operations on elements.
    """
    def __init__(self, n):
        """
        A method to create the union-find structure.
        Args:
            n (int): number of elements
        """
        self.parent = list(range(n))

    def find(self, i):
        """
        A method to find the root of the set that element 'i' belongs to.
        """
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        A method to perform the union of the sets that contain elements 'i' and 'j'
        """
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

    def getComponents(self):
        """
        A method to list all disjoint sets in the current structure.
        """
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return list(comps.values())
