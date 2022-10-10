#!/usr/bin/env python
import argparse
import os
import time
import pickle
import json
from tvm import auto_scheduler
from src.scripts.utils import get_device
from src.models.load_model import load_tvm_model
from src.data.transfer_tuning_utils import get_workload_info


def get_model_info(
    mod,
    params,
    device_info,
):
    tasks, _ = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        device_info["target"],
        device_info["host"],
    )
    wkl_classes, _ = get_workload_info(tasks)

    return (wkl_classes,)


def main(args):
    device_info = get_device(args.device_name)

    model_set_info_file = os.path.join(args.network_dir, "models_data.json")
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)

    for i, (m, model_info) in enumerate(model_set_info.items()):
        print("Generating task info for model", m, f"{i+1}/{len(model_set_info)}")
        relay_file, relay_params = model_info["fnames"]
        mod, params = load_tvm_model(relay_file, relay_params, args.network_dir)

        wkl_classes_tuning = get_model_info(mod, params, device_info)

        save_name = os.path.join(args.output_dir, m + "_wkl_classes.pkl")
        with open(save_name, "wb") as f:
            pickle.dump(wkl_classes_tuning, f)
    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Save the workload class info of models to file
        """
    )
    parser.add_argument(
        "--network_dir",
        type=str,
        required=True,
        help="Path to model files to use",
    )
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store tuned config files",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.network_dir
    start = time.time()
    main(args)
    end = time.time()
    print("Took:", end - start)
