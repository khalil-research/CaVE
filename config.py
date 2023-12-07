#!/usr/bin/env python
# coding: utf-8
"""
Training configuration
"""

from types import SimpleNamespace

# init configs
configs = {}

### ========================= SP5 =========================
configs["sp5"] = {}

# 2-stage
configs["sp5"]["2s"] = SimpleNamespace()
configs["sp5"]["2s"].lr = 1e-2
configs["sp5"]["2s"].epochs = 20

### ========================= TSP20 =========================
configs["tsp20"] = {}

# 2-stage
configs["tsp20"]["2s"] = SimpleNamespace()
configs["tsp20"]["2s"].lr = 5e-2
configs["tsp20"]["2s"].epochs = 20
