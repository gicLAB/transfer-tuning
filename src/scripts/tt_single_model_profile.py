#!/usr/bin/env python
import argparse
import os
from src.scripts.utils import get_device, setup_tt_logger, get_model_info
from src.data.tt_utils import (
    evaluate_mapping_v2,
    get_tracing_names_2,
    get_wkl_profiling_inf_time,
)


def main(args):
    device_info = get_device(args.device_name)
    logger = setup_tt_logger(f"data/scratchpad/profile_{args.model_name}.log")

    print(f"Running profiler for {args.model_name}")

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

    profiling_dir = "/tmp/tvmdbg"
    med_time, std_time = evaluate_mapping_v2(
        mod,
        params,
        test_inputs,
        target_outputs,
        {},
        None,
        args.model_name,
        args.model_path,
        args.device_name,
        profiling=True,
        runs=10,
        profiling_dir=profiling_dir,
    )
    prof_dir = os.path.join(profiling_dir, "_tvmdbg_device_CPU_0")
    # profiling_names = get_tracing_names_2(
    #     prof_dir, wkl_full_params, tasks, task_weights
    # )
    model_wkl_ids = list(wkl_classes.keys())
    tuned_wkl_inf_times = get_wkl_profiling_inf_time(
        model_wkl_ids,
        wkl_full_params,
        tasks,
        task_weights,
        prof_dir,
        drop_contrib_NCHWc=True,
    )
    print(tuned_wkl_inf_times)
    print(wkl_names)
    ret = {}
    for wkl_id, wkl_name in wkl_names.items():
        ret[wkl_name] = tuned_wkl_inf_times[wkl_id]
    print(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Get a profiling of the untuned workloads
        """
    )

    parser.add_argument("--model_path", type=str, required=True, help="Path to models")
    parser.add_argument("--model_name", type=str, required=True, help="Network to use")
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )
    args = parser.parse_args()
    main(args)
