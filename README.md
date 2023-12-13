# CaVE: Cone-Aligned Vector Estimation

<p align="center"><img width="50%" src="images/cone.png" /></p>

## Publication

This repository is the implementation of our paper. CaVE: A Cone-Aligned Approach for Fast Predict-then-optimize with Binary Linear Programs.

## Introduction

CaVE (Cone-aligned Vector Estimation) is a cutting-edge machine learning methodology designed for the efficient and accurate solution of predict-then-optimize tasks within the domain of operations research. This innovative approach integrates predictive modeling with optimization algorithms, focusing on Binary Linear Programming (BLP) problems.

## Key Features

- **End-to-End Training:** CaVE seamlessly integrates learning and optimization, delivering state-of-the-art performance in predicting cost coefficients for optimization problems.
- **Innovative Alignment Strategy:** By aligning predicted cost vectors within a cone, CaVE simplifies the original problem into more manageable quadratic programming.
- **Versatility and Scalability:** Exceptionally robust in managing large-scale optimization challenges, CaVE is adaptable to various problem sizes and complexities.
- **Efficiency in Computation:** Markedly improves computational efficiency, streamlining the traditionally more time-consuming and complex end-to-end predict-then-optimize approaches.

## Dependencies

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Pathos](https://pathos.readthedocs.io/)
* [tqdm](https://tqdm.github.io/)
* [cvxpy](https://www.cvxpy.org/)
* [Clarabel](https://oxfordcontrol.github.io/ClarabelDocs)
* [Gurobi](https://www.gurobi.com/)
* [PyTorch](http://pytorch.org/)
* [PyEPO](https://github.com/khalil-research/PyEPO)
