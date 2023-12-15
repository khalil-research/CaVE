#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments
"""

import argparse
import itertools
import os
import sys
sys.path.append("~/projects/def-khalile2/botang/caves/")

import submitit
import numpy as np

from config import hparams
from pipeline import pipeline

# job submission parameters
instance_logs_path = "slurm_logs_cavetest"
mem_gb = 16
num_cpus = 8
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--expnum",
                        type=int,
                        default=10,
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
                        choices=["sp5", "tsp20", "tsp50", "vrp20"],
                        help="problem type")

    # get experiment setting
    setting = parser.parse_args()

    # more mem
    if (setting.prob == "tsp50") or (setting.prob == "vrp20"):
        mem_gb = 32

    # config settings with hyperparameters changing
    confset_cavep = {"max_iter":range(1, 6)}
    confset_caveh = {"solve_ratio":np.arange(0.0, 1.0, 0.1),
                     "inner_ratio":np.arange(0.1, 1.0, 0.1)}

    # init job list
    jobs = []

    ############################################################################
    print("Hyperparameters for CaVE+...")
    setting.mthd = "cave+"
    # time out
    timeout_min = hparams[setting.prob][setting.mthd].timeout_min
    timeout_min *= setting.expnum
    # create executor
    executor = submitit.AutoExecutor(folder=instance_logs_path)
    executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                               timeout_min=timeout_min,
                               mem_gb=mem_gb,
                               cpus_per_task=num_cpus)
    for max_iter, in itertools.product(*tuple(confset_cavep.values())):
        # get exp setting
        print("Experiment setting:")
        print(setting)
        # get config
        hparams[setting.prob][setting.mthd].max_iter = max_iter
        print("Hyperparameters:")
        print(hparams[setting.prob][setting.mthd])
        # res dir
        res_dir = "./res/hparams/miter{}".format(max_iter)
        # run job
        job = executor.submit(pipeline, setting, hparams, res_dir)
        jobs.append(job)
        print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}".
        format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))
        print()

    ###########################################################################
    print("Hyperparameters for CaVE+ Hybrid...")
    setting.mthd = "caveh"
    # time out
    timeout_min = hparams[setting.prob][setting.mthd].timeout_min
    timeout_min *= setting.expnum
    # create executor
    executor = submitit.AutoExecutor(folder=instance_logs_path)
    executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                               timeout_min=timeout_min,
                               mem_gb=mem_gb,
                               cpus_per_task=num_cpus)
    for solve_ratio, inner_ratio in itertools.product(*tuple(confset_caveh.values())):
        # get exp setting
        print("Experiment setting:")
        print(setting)
        # get config
        hparams[setting.prob][setting.mthd].solve_ratio = solve_ratio
        hparams[setting.prob][setting.mthd].inner_ratio = inner_ratio
        print("Hyperparameters:")
        print(hparams[setting.prob][setting.mthd])
        # res dir
        res_dir = "./res/hparams/sratio{:.1f}_iratio{:.1f}".format(solve_ratio, inner_ratio)
        # run job
        job = executor.submit(pipeline, setting, hparams, res_dir)
        jobs.append(job)
        print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}".
        format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))
        print()

    # get outputs
    #outputs = [job.result() for job in jobs]
