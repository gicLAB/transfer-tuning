#!/usr/bin/env python
import os
import argparse
import json
from datetime import datetime
import pathlib
import shutil
import numpy as np
from tt_single_model_pact import main as run_tt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        For each workload in the new network, find the best tuned workload in the old logfile to use.
        Produces a new log file with the best version.
        """
    )
    parser.add_argument(
        "--split_log_file_dir", type=str, required=True, help="Presplit logfiles"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Files for network to use"
    )
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Testing purposes, do not save the output file",
    )
    parser.add_argument(
        "--disable_conv2d",
        action="store_true",
        help="Do not run transfer-tuning on the conv2d workoads",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs to take the average of"
    )
    args = parser.parse_args()

    tt_networks = {
        "resnet18": ["resnet50"],
        "resnet50": ["googlenet"],
        "alexnet": ["vgg16"],
        "vgg16": ["googlenet"],
        "mobilenetv2": ["efficentnetb4"],
        "googlenet": ["resnet50"],
        "mnasnet1_0": ["googlenet"],
        "efficentnetb0": ["efficentnetb4"],
        "efficentnetb4": ["efficentnetb0"],
        "bert-base-uncased-seq_class-256": ["mobilebert-base-uncased-seq_class-256"],
        "mobilebert-base-uncased-seq_class-256": ["bert-base-uncased-seq_class-256"],
    }

    # output_dir = os.path.join(
    #     "data/results/", datetime.now().strftime("tt_multi_models_%Y_%m_%d")
    # )
    output_dir = "data/results/tt_multi_models"
    p = pathlib.Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    for i, (model_name, tt_models) in enumerate(tt_networks.items()):
        results_file = os.path.join(output_dir, f"{model_name}.json")
        results = dict()
        args.model_name = model_name
        for j, tt_model_name in enumerate(tt_models):
            results[tt_model_name] = dict()
            best_times = []
            search_times = []
            for run in range(args.runs):

                f, k = 0, 0
                print(
                    f"Running {model_name} ({i+1}/{len(tt_networks)}), ",
                    f"with {tt_model_name} ({j+1}/{len(tt_models)}),",
                )

                # clear the `/tmp` directory, it seems elsewhere in the code
                # I am not cleaning up properly, and this could fill up the disk
                for p in pathlib.Path("/tmp/").glob("tmp*"):
                    shutil.rmtree(p)

                results[tt_model_name][run] = dict()

                args.tt_model_name = [tt_model_name]

                args.full_evaluations = f
                (
                    best_time,
                    best_wkl_inf_times,
                    mapping,
                    search_time,
                    _,
                    wkl_standalone_times,
                ) = run_tt(args)

                search_time = best_wkl_inf_times["standalone"]
                results[tt_model_name][run]["tt_time"] = best_time
                best_times.append(best_time)
                results[tt_model_name][run]["search_time"] = search_time
                search_times.append(search_time)
                results[tt_model_name][run]["best_wkl_inf_times"] = best_wkl_inf_times
                results[tt_model_name][run]["mapping"] = mapping
                results[tt_model_name][run][
                    "wkl_standalone_times"
                ] = wkl_standalone_times

            # collect the average values
            best_times = [b for b in best_times if b is not None]
            search_times = [b for b in search_times if b is not None]
            results[tt_model_name]["avg"] = dict()
            results[tt_model_name]["avg"]["tt_time"] = np.median(best_times)
            results[tt_model_name]["avg"]["search_time"] = np.median(search_times)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
