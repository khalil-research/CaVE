#!/usr/bin/env python
# coding: utf-8
"""
Experiment pipeline
"""

import os
import random

import numpy as np
import pandas as pd
import pyepo
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import collate_fn
from dataset import optDatasetConstrs
from pred import linearRegression
from train import train

from pyepo.model.grb import shortestPathModel
from model import tspDFJModel


def pipeline(config):
    # shortest path
    if config.prob[:2] == "sp":
        print("Running experiments for shortest path:")
    # travelling salesman
    if config.prob[:3] == "tsp":
        print("Running experiments for traveling salesman:")
    # get file path
    res_path = getDir(config.prob, config.mthd, config.data, config.deg)
    # create or load table
    if os.path.isfile(res_path): # exist res
        df = pd.read_csv(res_path)
    else:
        colnames = ["Train Regret", "Test Regret",
                    "Train MSE", "Test MSE",
                    "Train Elapsed", "Test Elapsed"]
        df = pd.DataFrame(columns=colnames)
    # build model
    optmodel = buildModel(config.prob)
    # num of experiments
    for i in range(config.expnum):
        # random seed for each experiment
        seed = 42 + i
        # set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # skip exist experiments
        if i < len(df):
            print("Skip experiment {}.".format(i))
            print(df.iloc[i:i+1])
            print()
            continue
        # start exp
        else:
            print("============================================================")
            print("Experiment {}:".format(i))
            print("============================================================")
        # generate data
        print("Generating synthetic data...")
        feats, costs = genData(config.prob, config.data, config.deg, seed)
        dataloaders = genDataLoader(optmodel, feats, costs, seed)
        # init predictor
        reg = linearRegression(feats.shape[1], costs.shape[1])
        # train and eval
        print("Training...")
        metrics = train(reg, optmodel, config.prob, config.mthd, *dataloaders)
        print("Evaluation:")
        print(metrics)
        # save data
        if df.empty:
            df = metrics
        else:
            df = pd.concat([df, metrics], ignore_index=True)
        df.to_csv(res_path, index=False)
        print()


def getDir(prob_name, mthd_name, num_data, poly_deg):
    """
    A method to get file path of csv result
    """
    # results
    res_dir = "./res/{}/n{}deg{}".format(prob_name, num_data, poly_deg)
    os.makedirs(res_dir, exist_ok=True)
    res_path = res_dir + "/{}.csv".format(mthd_name)
    return res_path


def buildModel(prob_name):
    """
    A method to build optimization model
    """
    # SP5
    if config.prob == "sp5":
        optmodel = shortestPathModel(grid=(5,5))
    # TSP20
    if config.prob == "tsp20":
        optmodel = tspDFJModel(num_nodes=20)
    # TSP50
    if config.prob == "tsp50":
        optmodel = tspDFJModel(num_nodes=50)
    return optmodel


def genDataLoader(optmodel, feats, costs, seed):
    """
    A method to get data loaders with solving optimal solutions
    """
    # data split
    x_train, x_test, c_train, c_test = train_test_split(feats, costs,
                                                        test_size=1000,
                                                        random_state=135)
    # create dataset
    dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
    dataset_train_ctr = optDatasetConstrs(optmodel, x_train, c_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
    # get data loader
    batch_size = 32
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_train_ctr = DataLoader(dataset_train_ctr, batch_size=batch_size,
                                  collate_fn=collate_fn, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_train_ctr, loader_test


def genData(prob_name, num_data, poly_deg, seed):
    """
    generate synthetic data
    """
    # SP5
    if config.prob == "sp5":
        feats, costs = pyepo.data.shortestpath.genData(num_data=num_data+1000,
                                                      num_features=5,
                                                      grid=(5,5),
                                                      deg=poly_deg,
                                                      noise_width=0.5,
                                                      seed=seed)
    # TSP20
    if config.prob == "tsp20":
        feats, costs = pyepo.data.tsp.genData(num_data=num_data+1000,
                                              num_features=5,
                                              num_nodes=20,
                                              deg=poly_deg,
                                              noise_width=0.5,
                                              seed=seed)
    if config.prob == "tsp50":
        feats, costs = pyepo.data.tsp.genData(num_data=num_data+1000,
                                              num_features=5,
                                              num_nodes=50,
                                              deg=poly_deg,
                                              noise_width=0.5,
                                              seed=seed)
    return feats, costs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--mthd",
                        type=str,
                        default="2s",
                        choices=["2s", "cave", "cave+", "caveh", "spo+", "pfyl", "nce"],
                        help="method")
    parser.add_argument("--expnum",
                        type=int,
                        default=1,
                        help="number of experiments")

    # data configuration
    parser.add_argument("--data",
                        type=int,
                        default=1000,
                        help="training data size")
    parser.add_argument("--deg",
                        type=int,
                        default=4,
                        help="features polynomial degree")

    # optimization model configuration
    parser.add_argument("--prob",
                        type=str,
                        default="sp5",
                        choices=["sp5", "tsp20", "tsp50"],
                        help="problem type")

    # get configuration
    config = parser.parse_args()

    # run experiment pipeline
    pipeline(config)
