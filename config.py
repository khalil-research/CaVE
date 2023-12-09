#!/usr/bin/env python
# coding: utf-8
"""
Training configuration
"""

from types import SimpleNamespace

# init hparams
hparams = {}

### ========================= SP5 =========================
hparams["sp5"] = {}

# 2-stage
hparams["sp5"]["2s"] = SimpleNamespace()
hparams["sp5"]["2s"].lr = 1e-2
hparams["sp5"]["2s"].epochs = 20
hparams["sp5"]["2s"].timeout_min = 2

# CaVE
hparams["sp5"]["cave"] = SimpleNamespace()
hparams["sp5"]["cave"].lr = 1e-2
hparams["sp5"]["cave"].epochs = 10
hparams["sp5"]["cave"].solver = "nnls"
hparams["sp5"]["cave"].timeout_min = 3

# CaVE+
hparams["sp5"]["cave+"] = SimpleNamespace()
hparams["sp5"]["cave+"].lr = 1e-2
hparams["sp5"]["cave+"].epochs = 10
hparams["sp5"]["cave+"].solver = "nnls"
hparams["sp5"]["cave+"].max_iter = None
hparams["sp5"]["cave+"].timeout_min = 3

# CaVE Hybrid
hparams["sp5"]["caveh"] = SimpleNamespace()
hparams["sp5"]["caveh"].lr = 1e-2
hparams["sp5"]["caveh"].epochs = 10
hparams["sp5"]["caveh"].solver = "nnls"
hparams["sp5"]["caveh"].solve_ratio = 0.3
hparams["sp5"]["caveh"].inner_ratio = 0.2
hparams["sp5"]["caveh"].timeout_min = 3

# SPO+
hparams["sp5"]["spo+"] = SimpleNamespace()
hparams["sp5"]["spo+"].lr = 1e-2
hparams["sp5"]["spo+"].epochs = 10
hparams["sp5"]["spo+"].timeout_min = 6

# PFYL
hparams["sp5"]["pfyl"] = SimpleNamespace()
hparams["sp5"]["pfyl"].lr = 1e-2
hparams["sp5"]["pfyl"].epochs = 10
hparams["sp5"]["pfyl"].n_samples = 1
hparams["sp5"]["pfyl"].sigma = 1.0
hparams["sp5"]["pfyl"].timeout_min = 5

# NCE
hparams["sp5"]["nce"] = SimpleNamespace()
hparams["sp5"]["nce"].lr = 1e-2
hparams["sp5"]["nce"].epochs = 20
hparams["sp5"]["nce"].solve_ratio = 0.05
hparams["sp5"]["nce"].timeout_min = 3

### ========================= TSP20 =========================
hparams["tsp20"] = {}

# 2-stage
hparams["tsp20"]["2s"] = SimpleNamespace()
hparams["tsp20"]["2s"].lr = 5e-2
hparams["tsp20"]["2s"].epochs = 20
hparams["tsp20"]["2s"].timeout_min = 3

# CaVE
hparams["tsp20"]["cave"] = SimpleNamespace()
hparams["tsp20"]["cave"].lr = 5e-2
hparams["tsp20"]["cave"].epochs = 10
hparams["tsp20"]["cave"].solver = "clarabel"
hparams["tsp20"]["cave"].timeout_min = 5

# CaVE+
hparams["tsp20"]["cave+"] = SimpleNamespace()
hparams["tsp20"]["cave+"].lr = 5e-2
hparams["tsp20"]["cave+"].epochs = 10
hparams["tsp20"]["cave+"].solver = "clarabel"
hparams["tsp20"]["cave+"].max_iter = 3
hparams["tsp20"]["cave+"].timeout_min = 5

# CaVE Hybrid
hparams["tsp20"]["caveh"] = SimpleNamespace()
hparams["tsp20"]["caveh"].lr = 5e-2
hparams["tsp20"]["caveh"].epochs = 10
hparams["tsp20"]["caveh"].solver = "clarabel"
hparams["tsp20"]["caveh"].solve_ratio = 0.3
hparams["tsp20"]["caveh"].inner_ratio = 0.2
hparams["tsp20"]["caveh"].timeout_min = 3

# SPO+
hparams["tsp20"]["spo+"] = SimpleNamespace()
hparams["tsp20"]["spo+"].lr = 5e-2
hparams["tsp20"]["spo+"].epochs = 10
hparams["tsp20"]["spo+"].timeout_min = 9

# PFYL
hparams["tsp20"]["pfyl"] = SimpleNamespace()
hparams["tsp20"]["pfyl"].lr = 5e-2
hparams["tsp20"]["pfyl"].epochs = 10
hparams["tsp20"]["pfyl"].n_samples = 1
hparams["tsp20"]["pfyl"].sigma = 1.0
hparams["tsp20"]["pfyl"].timeout_min = 7

# NCE
hparams["tsp20"]["nce"] = SimpleNamespace()
hparams["tsp20"]["nce"].lr = 5e-2
hparams["tsp20"]["nce"].epochs = 20
hparams["tsp20"]["nce"].solve_ratio = 0.05
hparams["tsp20"]["nce"].timeout_min = 5


### ========================= TSP50 =========================
hparams["tsp50"] = {}

# 2-stage
hparams["tsp50"]["2s"] = SimpleNamespace()
hparams["tsp50"]["2s"].lr = 5e-2
hparams["tsp50"]["2s"].epochs = 20
hparams["tsp50"]["2s"].timeout_min = 30

# CaVE
hparams["tsp50"]["cave"] = SimpleNamespace()
hparams["tsp50"]["cave"].lr = 5e-2
hparams["tsp50"]["cave"].epochs = 10
hparams["tsp50"]["cave"].solver = "clarabel"
hparams["tsp50"]["cave"].timeout_min = 30

# CaVE+
hparams["tsp50"]["cave+"] = SimpleNamespace()
hparams["tsp50"]["cave+"].lr = 5e-2
hparams["tsp50"]["cave+"].epochs = 10
hparams["tsp50"]["cave+"].solver = "clarabel"
hparams["tsp50"]["cave+"].max_iter = 3
hparams["tsp50"]["cave+"].timeout_min = 30

# CaVE Hybrid
hparams["tsp50"]["caveh"] = SimpleNamespace()
hparams["tsp50"]["caveh"].lr = 5e-2
hparams["tsp50"]["caveh"].epochs = 10
hparams["tsp50"]["caveh"].solver = "clarabel"
hparams["tsp50"]["caveh"].solve_ratio = 0.3
hparams["tsp50"]["caveh"].inner_ratio = 0.2
hparams["tsp50"]["caveh"].timeout_min = 30

# SPO+
hparams["tsp50"]["spo+"] = SimpleNamespace()
hparams["tsp50"]["spo+"].lr = 5e-2
hparams["tsp50"]["spo+"].epochs = 10
hparams["tsp50"]["spo+"].timeout_min = 50

# PFYL
hparams["tsp50"]["pfyl"] = SimpleNamespace()
hparams["tsp50"]["pfyl"].lr = 5e-2
hparams["tsp50"]["pfyl"].epochs = 10
hparams["tsp50"]["pfyl"].n_samples = 1
hparams["tsp50"]["pfyl"].sigma = 1.0
hparams["tsp50"]["pfyl"].timeout_min = 40

# NCE
hparams["tsp50"]["nce"] = SimpleNamespace()
hparams["tsp50"]["nce"].lr = 5e-2
hparams["tsp50"]["nce"].epochs = 20
hparams["tsp50"]["nce"].solve_ratio = 0.05
hparams["tsp50"]["nce"].timeout_min = 30


### ========================= VRP20 =========================
hparams["vrp20"] = {}

# 2-stage
hparams["vrp20"]["2s"] = SimpleNamespace()
hparams["vrp20"]["2s"].lr = 5e-2
hparams["vrp20"]["2s"].epochs = 20
hparams["vrp20"]["2s"].timeout_min = 50

# CaVE
hparams["vrp20"]["cave"] = SimpleNamespace()
hparams["vrp20"]["cave"].lr = 5e-2
hparams["vrp20"]["cave"].epochs = 10
hparams["vrp20"]["cave"].solver = "clarabel"
hparams["vrp20"]["cave"].timeout_min = 50

# CaVE+
hparams["vrp20"]["cave+"] = SimpleNamespace()
hparams["vrp20"]["cave+"].lr = 5e-2
hparams["vrp20"]["cave+"].epochs = 10
hparams["vrp20"]["cave+"].solver = "clarabel"
hparams["vrp20"]["cave+"].max_iter = 3
hparams["vrp20"]["cave+"].timeout_min = 50

# CaVE Hybrid
hparams["vrp20"]["caveh"] = SimpleNamespace()
hparams["vrp20"]["caveh"].lr = 5e-2
hparams["vrp20"]["caveh"].epochs = 10
hparams["vrp20"]["caveh"].solver = "clarabel"
hparams["vrp20"]["caveh"].solve_ratio = 0.3
hparams["vrp20"]["caveh"].inner_ratio = 0.2
hparams["vrp20"]["caveh"].timeout_min = 50

# SPO+
hparams["vrp20"]["spo+"] = SimpleNamespace()
hparams["vrp20"]["spo+"].lr = 5e-2
hparams["vrp20"]["spo+"].epochs = 10
hparams["vrp20"]["spo+"].timeout_min = 80

# PFYL
hparams["vrp20"]["pfyl"] = SimpleNamespace()
hparams["vrp20"]["pfyl"].lr = 5e-2
hparams["vrp20"]["pfyl"].epochs = 10
hparams["vrp20"]["pfyl"].n_samples = 1
hparams["vrp20"]["pfyl"].sigma = 1.0
hparams["vrp20"]["pfyl"].timeout_min = 60

# NCE
hparams["vrp20"]["nce"] = SimpleNamespace()
hparams["vrp20"]["nce"].lr = 5e-2
hparams["vrp20"]["nce"].epochs = 20
hparams["vrp20"]["nce"].solve_ratio = 0.05
hparams["vrp20"]["nce"].timeout_min = 50
