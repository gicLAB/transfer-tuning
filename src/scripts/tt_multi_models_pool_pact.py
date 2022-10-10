#!/usr/bin/env python
import os
import argparse
import json
from datetime import datetime
import pathlib
import shutil
import numpy as np
from tt_single_model_pool_pact import main as run_tt_pool


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
        # "resnet18": [
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "googlenet",
        #     "mnasnet1_0",
        #     # "efficentnetb0",
        #     # "efficentnetb4",
        # ],
        # "resnet50": [
        #     "resnet18",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "alexnet": [
        #     "resnet18",
        #     "resnet50",
        #     "mobilenetv2",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "vgg16": [
        #     "resnet18",
        #     "alexnet",
        #     "resnet50",
        #     "mobilenetv2",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "mobilenetv2": [
        #     "resnet18",
        #     "resnet50",
        #     "alexnet",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        # ],
        # "googlenet": [
        #     "resnet18",
        #     "alexnet",
        #     "mobilenetv2",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "mnasnet1_0": [
        #     "resnet18",
        #     "alexnet",
        #     "resnet50",
        #     "vgg16",
        #     "mobilenetv2",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "efficentnetb0": [
        #     "resnet18",
        #     "alexnet",
        #     "resnet50",
        #     "vgg16",
        #     "mobilenetv2",
        #     "mnasnet1_0",
        #     ""
        # ],
        # "efficentnetb4": [
        #     "resnet18",
        #     "alexnet",
        #     "resnet50",
        #     "vgg16",
        #     "mobilenetv2",
        #     "mnasnet1_0",
        # ],
    }

    tt_networks = {
        # "resnet18": [
        #     "resnet50",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "resnet50": [
        #     "resnet18",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "alexnet": [
        #     "resnet18",
        #     "resnet50",
        #     "mobilenetv2",
        #     "googlenet",
        #     "vgg16",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "vgg16": [
        #     "resnet18",
        #     "resnet50",
        #     "mobilenetv2",
        #     "googlenet",
        #     "alexnet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "mobilenetv2": [
        #     "resnet18",
        #     "resnet50",
        #     "vgg16",
        #     "googlenet",
        #     "alexnet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "googlenet": [
        #     "resnet18",
        #     "resnet50",
        #     "mobilenetv2",
        #     "vgg16",
        #     "alexnet",
        #     "mnasnet1_0",
        #     "efficentnetb0",
        #     "efficentnetb4",
        # ],
        # "mnasnet1_0": [
        #     "resnet18",
        #     "resnet50",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "efficentnetb0",
        #     "efficentnetb4",
        #     "googlenet",
        # ],
        "efficentnetb0": [
            #     "resnet18",
            #     "resnet50",
            #     "alexnet",
            #     "vgg16",
            #     "mobilenetv2",
            "efficentnetb4",
            #     "googlenet",
            #     "mnasnet1_0",
        ],
        # "efficentnetb4": [
        #     # "resnet18",
        #     # "resnet50",
        #     # "alexnet",
        #     # "vgg16",
        #     # "mobilenetv2",
        #     "efficentnetb0",
        #     # "googlenet",
        #     # "mnasnet1_0",
        # ],
        #
        # "mobilebert-base-uncased-seq_class-256": [
        #     "resnet18",
        #     "resnet50",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "efficentnetb0",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "bert-base-uncased-seq_class-256",
        # ],
        # "bert-base-uncased-seq_class-256": [
        #     "resnet18",
        #     "resnet50",
        #     "alexnet",
        #     "vgg16",
        #     "mobilenetv2",
        #     "efficentnetb0",
        #     "googlenet",
        #     "mnasnet1_0",
        #     "mobilebert-base-uncased-seq_class-256",
        # ],
    }

    output_dir = os.path.join(
        "data/results/", datetime.now().strftime("tt_multi_models_pool_%Y_%m_%d")
    )
    p = pathlib.Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)

    for i, (model_name, tt_models) in enumerate(tt_networks.items()):
        results_file = os.path.join(output_dir, f"{model_name}.json")
        results = dict()
        args.model_name = model_name

        for num_schedules in [1e10]:
            # for num_schedules in [1e10]:
            results[num_schedules] = dict()
            best_times = []
            search_times = []
            for run in range(args.runs):
                results[num_schedules][run] = dict()

                f, k = 0, 0
                # print(
                #     f"Running {model_name} ({i+1}/{len(tt_networks)}), ",
                #     f"with {tt_model_name} ({j+1}/{len(tt_models)}),",
                # )

                # clear the `/tmp` directory, it seems elsewhere in the code
                # I am not cleaning up properly, and this could fill up the disk
                for p in pathlib.Path("/tmp/").glob("tmp*"):
                    shutil.rmtree(p)

                results[num_schedules][run] = dict()

                args.tt_model_name = tt_models
                args.num_schedules = num_schedules

                args.full_evaluations = f
                (
                    best_time,
                    best_wkl_inf_times,
                    mapping,
                    search_time,
                    untuned_time,
                    wkl_standalone_times,
                ) = run_tt_pool(args)

                search_time = best_wkl_inf_times["standalone"]
                results[num_schedules][run]["tt_time"] = best_time
                best_times.append(best_time)
                results[num_schedules][run]["search_time"] = search_time
                search_times.append(search_time)
                results[num_schedules][run]["untuned_time"] = untuned_time
                results[num_schedules][run]["best_wkl_inf_times"] = best_wkl_inf_times
                results[num_schedules][run]["mapping"] = mapping
                results[num_schedules][run][
                    "wkl_standalone_times"
                ] = wkl_standalone_times

            # collect the average values
            best_times = [b for b in best_times if b is not None]
            search_times = [b for b in search_times if b is not None]
            results[num_schedules]["best_times"] = best_times
            results[num_schedules]["search_times"] = search_times
            results[num_schedules]["avg"] = dict()
            results[num_schedules]["avg"]["tt_time"] = np.median(best_times)
            results[num_schedules]["avg"]["search_time"] = np.median(search_times)
            # print(m, best_times)
            if not args.test:
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
