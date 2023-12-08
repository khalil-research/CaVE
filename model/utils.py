#!/usr/bin/env python
# coding: utf-8
"""
Utilities
"""

from collections import defaultdict

class unionFind:
    """
    Union-find disjoint sets
    """
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False

    def getComponents(self):
        comps = defaultdict(list)
        for i in range(len(self.parent)):
            comps[self.find(i)].append(i)
        return list(comps.values())
