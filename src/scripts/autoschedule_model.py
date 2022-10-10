#!/usr/bin/env python
import os
import argparse
import json
from datetime import datetime
from typing import Union, Dict

from src.scripts.utils import get_device
from src.data.autosched_utils import tune_and_evaluate, compile_tuned_graph
from src.inference.tvm_inference import evaluate_untuned_ansor
from src.models.load_model import load_tvm_model


def tune_model(
    model_name,
    model_path,
    output_dir,
    device_info,
    device_name,
    ntrials,
    output_csv_file,
    timeout: int,
    tt_file: Union[os.PathLike, Dict] = None,
    finetune=False,
    stop_points=None,
    minute_check=False,
):

    batch_size = 1
    layout = "NCHW"
    target = device_info["target"]

    model_set_info_file = os.path.join(model_path, "models_data.json")
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)
    model_info = model_set_info[model_name]

    if tt_file:
        # when we have a set of search times from an experiment
        with open(tt_file) as f:
            # get the first value
            data = json.load(f)
            key = list(data.keys())[0]

        if minute_check:
            stop_point = data[key]["avg"]["search_time"]
            stop_points = [stop_point] + [stop_point + i * 60 for i in range(1, 30)]
        else:
            evals = data[key].keys()
            search_times = {e: data[key][e]["search_time"] for e in evals}
            stop_points = list(search_times.values())
        # stop_points = [s / 10 for s in stop_points] # for debugging purposes only
        print("Hey, stop points are", stop_points)
        # exit(1)
    elif stop_points is not None:
        stop_points = stop_points
    else:
        stop_points = None

    # get baseline time
    relay_file, relay_params = model_info["fnames"]
    mod, params = load_tvm_model(relay_file, relay_params, model_path)
    print(f"Running {model_name} untuned")
    default_ansor_med, _ = evaluate_untuned_ansor(
        mod,
        params,
        device_info,
        model_info["test_input_data"],
        model_info["target_outputs"],
    )

    # do normal tuning
    mod, params = load_tvm_model(relay_file, relay_params, model_path)
    log_file = os.path.join(
        output_dir,
        "%s£-%s-B%d-%s-%s.json" % (model_name, layout, batch_size, target, device_name),
    )
    if finetune is False:
        # delete pre-existing tuning, so we have a fresh start
        try:
            os.remove(log_file)
        except:
            pass

    tuned_med, tuned_std, tuning_time = tune_and_evaluate(
        log_file,
        mod,
        params,
        model_info["test_input_data"],
        model_info["target_outputs"],
        device_info,
        ntrials=ntrials,
        timeout=timeout,
        stop_points=stop_points,
    )

    # if tt_file:
    #     if not minute_check:
    #         meds = dict()
    #         for i, (evals, search_time) in enumerate(search_times.items()):
    #             # print(f"Evaluating {i}: {evals}, {search_time}")
    #             log_file_e = log_file + f".{i}.json"
    #             mod, params = load_tvm_model(relay_file, relay_params, model_path)
    #             med, std = compile_tuned_graph(
    #                 log_file_e,
    #                 mod,
    #                 params,
    #                 model_info["test_input_data"],
    #                 model_info["target_outputs"],
    #                 device_info,
    #                 dtype="float32",
    #                 use_ndk=False,
    #             )
    #             print(f"Evaluating {i}: {evals}, {search_time}, inf time: {med}")
    #             meds[evals] = med
    #             data[key][str(evals)]["ansor_inf_time"] = med
    #         with open(tt_file, "w") as f:
    #             json.dump(data, f, indent=2)
    #         print(meds)
    if minute_check:
        meds = dict()
        for i, t in enumerate(stop_points):
            log_file_e = log_file + f".{i}.json"
            mod, params = load_tvm_model(relay_file, relay_params, model_path)
            med, std = compile_tuned_graph(
                log_file_e,
                mod,
                params,
                model_info["test_input_data"],
                model_info["target_outputs"],
                device_info,
                dtype="float32",
                use_ndk=False,
            )
            meds[t] = med
        sppath = "/workspace/data/results/tt_stop_points"
        os.makedirs(sppath, exist_ok=True)

        with open(os.path.join(sppath, model_name + ".json"), "w") as f:
            json.dump(meds, f, indent=2)
    return tuned_med, tuning_time, default_ansor_med


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune a set single model from scratch."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to tune"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path where models are stored"
    )
    parser.add_argument(
        "--ntrials", type=int, default="20000", help="Number of trials to tune for"
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Number of seconds before stopping tuning",
    )
    parser.add_argument(
        "--tt_file",
        type=str,
        default=None,
        help="Output from running TT",
    )
    parser.add_argument(
        "--stop_points",
        type=int,
        default=None,
        nargs="+",
        help="Stop points for making checkopoints of tuning",
    )
    parser.add_argument(
        "--finetune",
        dest="finetune",
        action="store_true",
        help="Not currently used, when set true will not delete existing tuning",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device_info = get_device(args.device_name)
    output_csv_file = os.path.join(args.output_dir, "tuning_info.csv")

    tuned_time, tuning_time = tune_model(
        args.model_name,
        args.model_path,
        args.output_dir,
        device_info,
        args.device_name,
        args.ntrials,
        output_csv_file,
        args.timeout,
        args.tt_file,
        args.finetune,
        stop_points=args.stop_points,
    )
