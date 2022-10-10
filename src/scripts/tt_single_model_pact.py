#!/usr/bin/env python
import argparse
import os
import time
import pickle
from src.scripts.utils import get_device, get_model_info, setup_tt_logger
from src.data.transfer_tuning_utils_2 import (
    tt_internetwork_main_2,
)
from src.inference.tvm_inference import evaluate_untuned_ansor


def main(args):
    device_info = get_device(args.device_name)
    # logger = setup_tt_logger(f"data/scratchpad/profile_{args.model_name}.log")

    print(f"Running inter-network oracle for {args.model_name})")

    (
        mod,
        params,
        test_inputs,
        target_outputs,
        tasks,
        wkl_classes,
        full_wkl_ids,
        task_weights,
        wkl_names,
        wkl_full_params,
    ) = get_model_info(args.model_name, args.model_path, device_info)
    print(wkl_names)
    wkl_classes_tuning = dict()
    for tt_model_name in args.tt_model_name:
        with open(
            os.path.join(args.model_path, tt_model_name + "_wkl_classes.pkl"), "rb"
        ) as f:
            new_wkl_classes_tuning = pickle.load(f)[0]
        wkl_classes_tuning = {**new_wkl_classes_tuning, **wkl_classes_tuning}

    print(len(wkl_classes_tuning))

    tuned_wkl_ids = list(wkl_classes_tuning.keys())
    start = time.time()

    (
        best_time,
        best_wkl_inf_times,
        mapping,
        wkl_standalone_times,
    ) = tt_internetwork_main_2(
        mod,
        params,
        test_inputs,
        target_outputs,
        args.device_name,
        tasks,
        wkl_classes,
        wkl_classes_tuning,
        wkl_names,
        full_wkl_ids,
        args.model_name,
        args.model_path,
        args.split_log_file_dir,
        task_weights=task_weights,
        wkl_full_params=wkl_full_params,
        disable_conv2d=args.disable_conv2d,
    )
    end = time.time()
    search_time = end - start
    print(
        f"{args.model_name}:",
        f"Search time: {search_time},",
        f"best_time: {best_time}",
    )
    return (
        best_time,
        best_wkl_inf_times,
        mapping,
        search_time,
        -1,
        wkl_standalone_times,
    )


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
    parser.add_argument("--model_path", type=str, required=True, help="Path to models")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of network to use"
    )
    parser.add_argument(
        "--tt_model_name",
        type=str,
        required=True,
        nargs="+",
        help="Name of model(s) to use for transfer-tuning",
    )
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )

    parser.add_argument(
        "--disable_conv2d",
        action="store_true",
        help="Do not run transfer-tuning on the conv2d workoads",
    )
    args = parser.parse_args()
    main(args)
