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

# CaVE
configs["sp5"]["cave"] = SimpleNamespace()
configs["sp5"]["cave"].lr = 1e-2
configs["sp5"]["cave"].epochs = 10
configs["sp5"]["cave"].solver = "nnls"

# CaVE+
configs["sp5"]["cave+"] = SimpleNamespace()
configs["sp5"]["cave+"].lr = 1e-2
configs["sp5"]["cave+"].epochs = 10
configs["sp5"]["cave+"].solver = "nnls"

# CaVE Hybrid
configs["sp5"]["caveh"] = SimpleNamespace()
configs["sp5"]["caveh"].lr = 1e-2
configs["sp5"]["caveh"].epochs = 10
configs["sp5"]["caveh"].solver = "nnls"
configs["sp5"]["caveh"].solve_ratio = 0.3
configs["sp5"]["caveh"].inner_ratio = 0.2

# SPO+
configs["sp5"]["spo+"] = SimpleNamespace()
configs["sp5"]["spo+"].lr = 1e-2
configs["sp5"]["spo+"].epochs = 10

# PFYL
configs["sp5"]["pfyl"] = SimpleNamespace()
configs["sp5"]["pfyl"].lr = 1e-2
configs["sp5"]["pfyl"].epochs = 10
configs["sp5"]["pfyl"].n_samples = 1
configs["sp5"]["pfyl"].sigma = 1.0

# NCE
configs["sp5"]["nce"] = SimpleNamespace()
configs["sp5"]["nce"].lr = 1e-2
configs["sp5"]["nce"].epochs = 10
configs["sp5"]["nce"].solve_ratio = 0.05


### ========================= TSP20 =========================
configs["tsp20"] = {}

# 2-stage
configs["tsp20"]["2s"] = SimpleNamespace()
configs["tsp20"]["2s"].lr = 5e-2
configs["tsp20"]["2s"].epochs = 20

# CaVE
configs["tsp20"]["cave"] = SimpleNamespace()
configs["tsp20"]["cave"].lr = 5e-2
configs["tsp20"]["cave"].epochs = 10
configs["tsp20"]["cave"].solver = "clarabel"

# CaVE+
configs["tsp20"]["cave+"] = SimpleNamespace()
configs["tsp20"]["cave+"].lr = 5e-2
configs["tsp20"]["cave+"].epochs = 10
configs["tsp20"]["cave+"].solver = "clarabel"

# CaVE Hybrid
configs["tsp20"]["caveh"] = SimpleNamespace()
configs["tsp20"]["caveh"].lr = 5e-2
configs["tsp20"]["caveh"].epochs = 10
configs["tsp20"]["caveh"].solver = "clarabel"
configs["tsp20"]["caveh"].solve_ratio = 0.3
configs["tsp20"]["caveh"].inner_ratio = 0.2

# SPO+
configs["tsp20"]["spo+"] = SimpleNamespace()
configs["tsp20"]["spo+"].lr = 5e-2
configs["tsp20"]["spo+"].epochs = 10

# PFYL
configs["tsp20"]["pfyl"] = SimpleNamespace()
configs["tsp20"]["pfyl"].lr = 5e-2
configs["tsp20"]["pfyl"].epochs = 10
configs["tsp20"]["pfyl"].n_samples = 1
configs["tsp20"]["pfyl"].sigma = 1.0

# NCE
configs["tsp20"]["nce"] = SimpleNamespace()
configs["tsp20"]["nce"].lr = 5e-2
configs["tsp20"]["nce"].epochs = 10
configs["tsp20"]["nce"].solve_ratio = 0.05


### ========================= TSP50 =========================
configs["tsp50"] = {}

# 2-stage
configs["tsp50"]["2s"] = SimpleNamespace()
configs["tsp50"]["2s"].lr = 5e-2
configs["tsp50"]["2s"].epochs = 20

# CaVE
configs["tsp50"]["cave"] = SimpleNamespace()
configs["tsp50"]["cave"].lr = 5e-2
configs["tsp50"]["cave"].epochs = 10
configs["tsp50"]["cave"].solver = "clarabel"

# CaVE+
configs["tsp50"]["cave+"] = SimpleNamespace()
configs["tsp50"]["cave+"].lr = 5e-2
configs["tsp50"]["cave+"].epochs = 10
configs["tsp50"]["cave+"].solver = "clarabel"

# CaVE Hybrid
configs["tsp50"]["caveh"] = SimpleNamespace()
configs["tsp50"]["caveh"].lr = 5e-2
configs["tsp50"]["caveh"].epochs = 10
configs["tsp50"]["caveh"].solver = "clarabel"
configs["tsp50"]["caveh"].solve_ratio = 0.3
configs["tsp50"]["caveh"].inner_ratio = 0.2

# SPO+
configs["tsp50"]["spo+"] = SimpleNamespace()
configs["tsp50"]["spo+"].lr = 5e-2
configs["tsp50"]["spo+"].epochs = 10

# PFYL
configs["tsp50"]["pfyl"] = SimpleNamespace()
configs["tsp50"]["pfyl"].lr = 5e-2
configs["tsp50"]["pfyl"].epochs = 10
configs["tsp50"]["pfyl"].n_samples = 1
configs["tsp50"]["pfyl"].sigma = 1.0

# NCE
configs["tsp50"]["nce"] = SimpleNamespace()
configs["tsp50"]["nce"].lr = 5e-2
configs["tsp50"]["nce"].epochs = 10
configs["tsp50"]["nce"].solve_ratio = 0.05