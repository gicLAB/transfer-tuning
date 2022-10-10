#!/usr/bin/env python
import argparse
import os
import time
import json
from tvm import auto_scheduler
from src.scripts.utils import get_device
from src.models.load_model import load_tvm_model

from src.data.tt_utils import get_wkl_profiling_inf_time
from src.data.workload_utils import get_workload_info_and_params
from src.inference.tvm_inference import evaluate_untuned_ansor


def get_model_info(
    mod,
    params,
    device_info,
):

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        device_info["target"],
        device_info["host"],
    )
    # (
    #     wkl_classes,
    #     full_wkl_ids,
    #     wkl_full_params,
    # ) = get_workload_info(tasks)

    (
        wkl_classes,
        full_wkl_ids,
        wkl_full_params,
        wkl_names,
    ) = get_workload_info_and_params(tasks, readable_names=True)
    return (
        tasks,
        task_weights,
        wkl_classes,
        full_wkl_ids,
        wkl_full_params,
        wkl_names,
    )


def main(args):
    device_info = get_device(args.device_name)
    model_set_info_file = os.path.join(args.model_path, "models_data.json")
    output_json = os.path.join(args.output_dir, "models_table.json")

    results = dict()
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)
    for i, (m, model_info) in enumerate(model_set_info.items()):
        print(f"Running {m}, {i}/{len(model_set_info)}")
        # if m != "resnet18":
        #     continue
        # if m != "resnet50":
        #     continue
        #
        # if m != "mobilenetv2":
        #     continue
        # if m != "mobilenetv3":
        #     continue
        # if m != "efficentnetb0":
        #     continue
        # if m == "efficentnetb4":
        #     continue
        # if m == "shufflenet_v2_x1_0":
        #     continue
        # if m != "mnasnet0_5":
        #     continue
        if "mobilebert" not in m:
            continue
        # if "mobile" not in m:
        #     continue
        if "shufflenet" in m:
            continue
        if "dcgan" in m:
            continue
        # if i <= 21:
        #     continue
        results[m] = dict()
        relay_file, relay_params = model_info["fnames"]
        mod, params = load_tvm_model(relay_file, relay_params, args.model_path)
        (
            tasks,
            task_weights,
            wkl_classes,
            full_wkl_ids,
            wkl_full_params,
            wkl_names,
        ) = get_model_info(mod, params, device_info)

        evaluate_untuned_ansor(
            mod,
            params,
            device_info,
            model_info["test_input_data"],
            model_info["target_outputs"],
            profile=True,
            runs=1,
        )
        exit(1)
        profiling_dir = "/tmp/tvmdbg"
        model_wkl_ids = list(wkl_classes.keys())
        wkl_inf_times = get_wkl_profiling_inf_time(
            model_wkl_ids,
            wkl_full_params,
            tasks,
            task_weights,
            os.path.join(profiling_dir, "_tvmdbg_device_CPU_0"),
        )
        results[m]["wkl_inf_times"] = wkl_inf_times
        results[m]["wkl_classes"] = wkl_classes
        results[m]["wkl_names"] = wkl_names
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\tCompleted {m}, it inference took a wee bit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Generate a table of information about the models and their workload
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to models files for network to use",
    )
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store tuned config files",
    )
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
