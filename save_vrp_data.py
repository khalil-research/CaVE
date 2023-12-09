#!/usr/bin/env python
# coding: utf-8
"""
Generate and save VRP optDataset
"""

import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pyepo

from model import vrpModel
from dataset import optDatasetConstrs

def saveVRPData(seed=42):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # save dir
    save_dir = "./data"
    # check if dir existed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # generate data
    num_node = 30 # node size
    num_data = 1000 # number of training data
    num_feat = 10 # size of feature
    deg = 4 # polynomial degree
    e = 0.5 # noise width
    capacity = 30 # vehicle capacity
    num_vehicle = 8 # number of vehicle
    feats, costs = pyepo.data.tsp.genData(num_data+1000, num_feat, num_node+1, deg, e, seed=seed)

    # set solver
    demands = np.random.rand(num_node) * 10 # demands
    optmodel = vrpModel(num_node+1, demands=demands, capacity=capacity, num_vehicle=num_vehicle)
    # check feasibility
    if np.sum(demands) >= capacity * num_vehicle:
        raise ValueError("Infeasible model.")
    # turn on output
    #optmodel._model.Params.outputFlag = 1
    # set time limit
    optmodel._model.Params.timelimit = 60

    # split data
    x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=1000, random_state=seed)

    # get training dataset
    dataset_train = optDatasetConstrs(optmodel, x_train, costs=c_train, skip_infeas=True)
    # save tensors
    torch.save(torch.FloatTensor(dataset_train.feats), save_dir+"/feats_train_vrp30.pt")
    torch.save(torch.FloatTensor(dataset_train.costs), save_dir+"/costs_train_vrp30.pt")
    torch.save(torch.FloatTensor(dataset_train.sols), save_dir+"/sols_train_vrp30.pt")
    torch.save(dataset_train.ctrs, save_dir+"/ctrs_train_vrp30.pt")

    # get test dataset
    dataset_test = optDatasetConstrs(optmodel, x_test, costs=c_test, skip_infeas=True)
    # save tensors
    torch.save(torch.FloatTensor(dataset_test.feats), save_dir+"/feats_test_vrp30.pt")
    torch.save(torch.FloatTensor(dataset_test.costs), save_dir+"/costs_test_vrp30.pt")
    torch.save(torch.FloatTensor(dataset_test.sols), save_dir+"/sols_test_vrp30.pt")
    torch.save(dataset_test.ctrs, save_dir+"/ctrs_test_vrp30.pt")

    # load for test
    #sols = torch.load(save_dir+"/sols_test_vrp30.pt")
    #print(sols.shape)
    #print(sols)


if __name__ == "__main__":

    import submitit

    # job submission parameters
    instance_logs_path = "slurm_logs_cavetest"
    mem_gb = 32
    num_cpus = 8
    timeout_min = 900
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)

    # create executor
    executor = submitit.AutoExecutor(folder=instance_logs_path)
    executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                               timeout_min=timeout_min,
                               mem_gb=mem_gb,
                               cpus_per_task=num_cpus)

    # run job
    job = executor.submit(saveVRPData)
    print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}".
    format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))
    print()

    # get outputs
    #job.result()
