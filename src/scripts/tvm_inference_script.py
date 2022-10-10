#!/usr/bin/env python
import argparse
import pickle
import os
import json

from src.scripts.utils import get_device, setup_tt_logger
from src.models.load_model import load_tvm_model
from src.inference.tvm_inference import evaluate_tuned_ansor, evaluate_untuned_ansor


def dump_pickle(save_name, data):
    with open(save_name, "wb") as f:
        pickle.dump(data, f)


def main(args, logger=None):
    import time

    timings = dict()
    start = time.time()
    device_info = get_device(args.device_name)

    model_set_info_file = os.path.join(args.model_path, "models_data.json")
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)
    model_info = model_set_info[args.model_name]
    relay_file, relay_params = model_info["fnames"]
    mod, params = load_tvm_model(relay_file, relay_params, args.model_path)
    test_inputs, target_outputs = (
        model_info["test_input_data"],
        model_info["target_outputs"],
    )
    end = time.time()
    print(f"It took {end - start} to load the model")
    timings["model_loading"] = end - start

    try:
        print(f"Trying to run inference for {args.model_name}")
        start = time.time()
        if args.log_file is not None:
            med, std = evaluate_tuned_ansor(
                args.log_file,
                mod,
                params,
                device_info,
                test_inputs,
                target_outputs,
                limit_std=args.limit_std,
                profile=args.profile,
                profile_dir=args.profile_dir,
            )
        else:
            print(f"Warning running untuned model {args.model_name}")
            med, std = evaluate_untuned_ansor(
                mod,
                params,
                device_info,
                test_inputs,
                target_outputs,
            )
        end = time.time()
        print(f"We ran inference for {args.model_name} and got {med}ms")
        if logger is not None:
            logger.info(f"We ran inference for {args.model_name} and got {med} ms")

        timings["model_inference"] = end - start
        print(timings)
        if args.output_file is not None:
            dump_pickle(args.output_file, (med, std))
    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a model, or set of models."
        "optionally include a given workload tuing map"
        "built because sometimes the TVM runtime gets corrupted, contains the issue"
    )
    parser.add_argument("--model_name", type=str, help="Name of the model to run")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path where model and relevant data is stored",
    )
    parser.add_argument(
        "--device_name", type=str, required=True, help="Device to run on"
    )
    parser.add_argument("--log_file", type=str, default=None, help="Log file to parse")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save output to",
    )
    parser.add_argument(
        "--limit_std",
        default=None,
        type=float,
        help="Re-run experiments that have a std that is too high",
    )
    parser.add_argument(
        "--runs",
        default=10,
        type=int,
        help="Number of times to run experriment",
    )
    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="data/scratchpad/tvmdbg",
        help="Directory to store profiling results, ignored if `--profile` is set to false",
    )
    args = parser.parse_args()

    if args.all_models is False:
        main(args)
    else:
        # run basic inference for every model in a model set
        assert args.log_file is None
        logger = setup_tt_logger(os.path.join("inference.log"))
        model_set_info_file = os.path.join(args.model_path, "models_data.json")
        with open(model_set_info_file) as f:
            model_set_info = json.load(f)
        models = list(model_set_info.keys())
        for i, m in enumerate(models):
            args.model_name = m
            print(f"Running {m}, model {i}/{len(models)}")
            try:
                main(args, logger)
            except:
                logger.exception(f"error for model {m}")
                pass
    print("Ran all models")
