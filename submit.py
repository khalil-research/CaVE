#!/usr/bin/env python
# coding: utf-8
"""
Submit experiments
"""

import argparse
import os
import sys
sys.path.append("~/projects/def-khalile2/botang/caves/")

from config import configs
from pipeline import pipeline

# job submission parameters
instance_logs_path = "slurm_logs_cavetest"
mem_gb = 16
num_cpus = 8
os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)

# all methods
methods = ["2s", "cave", "cave+", "caveh", "spo+", "pfyl", "nce"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # experiments configuration
    parser.add_argument("--expnum",
                        type=int,
                        default=5,
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

    # get experiment setting
    setting = parser.parse_args()

    # submit
    jobs = []
    for mthd_name in methods:
        setting.mthd = mthd_name
        # get exp setting
        print("Experiment setting:")
        print(setting)
        # time out
        timeout_min = configs[setting.prob][setting.mthd].timeout_min
        # create executor
        executor = submitit.AutoExecutor(folder=instance_logs_path)
        executor.update_parameters(slurm_additional_parameters={"account": "rrg-khalile2"},
                                   timeout_min=timeout_min,
                                   mem_gb=mem_gb,
                                   cpus_per_task=num_cpus)
        # run job
        job = executor.submit(pipeline, setting)
        jobs.append(job)
        print("job_id: {}, mem_gb: {}, num_cpus: {}, logs: {}, timeout: {}".
        format(job.job_id, mem_gb, num_cpus, instance_logs_path, timeout_min))
        print()

    # get outputs
    #outputs = [job.result() for job in jobs]
