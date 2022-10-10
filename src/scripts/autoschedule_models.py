#!/usr/bin/env python
import os
import argparse
import json

from src.models.load_model import load_tvm_model
from src.scripts.utils import get_device, setup_tt_logger
from src.scripts.autoschedule_model import tune_model


def main(args):
    # create output dir name by default
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = setup_tt_logger(os.path.join(args.output_dir, "tuning.log"))

    device_info = get_device(args.device_name)
    output_csv_file = os.path.join(args.output_dir, "tuning_info.csv")
    json_file = os.path.join(args.output_dir, "tuning_info.json")
    results = dict()

    model_set_info_file = os.path.join(args.model_path, "models_data.json")
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)

    skip_models = []

    for i, (model_name, model_info) in enumerate(model_set_info.items()):
        if model_name in skip_models:
            continue
        logger.info(
            f"Tuning {model_name} ({i}/{len(model_set_info) - len(skip_models)})"
        )
        relay_file, relay_params = model_info["fnames"]

        # get the path to the TT tuning file to use if specified
        if args.tt_path is None:
            tt_file = None
        else:
            # is there a better way than undocumented implicit file names? 🤔
            tt_file = os.path.join(args.tt_path, model_name + ".json")
            # with open(tt_file) as f:
            #     data = json.load(f)
            # assert (
            #     len(data.keys()) == 1
            # )  # not covering the case for several options right now
            # tt_data = list(data.values())[0]
            # get first value
        try:
            tuned_time, tuning_time, untuned_time = tune_model(
                model_name,
                args.model_path,
                args.output_dir,
                device_info,
                args.device_name,
                args.ntrials,
                output_csv_file,
                args.timeout,
                tt_file,
                minute_check=args.minute_check,
            )
            results[model_name] = {
                "tuned_time": tuned_time,
                "tuning_time": tuning_time,
                "untuned_time": untuned_time,
            }
            logger.info(
                f"{model_name}: tuned_time: {tuned_time}, tuning_time: {tuning_time}"
            )
        except Exception as e:
            logger.exception(f"error for model {model_name}")
            pass

        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune a set of models from scratch.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="ONNX file for network to use"
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
        default=None,
        help="Directory to store tuned config files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Number of seconds before stopping tuning",
    )
    parser.add_argument(
        "--tt_path",
        type=str,
        default=None,
        help="If set, will use the search time from running TT, used to limit tuning time",
    )
    parser.add_argument("--minute_check", dest="minute_check", action="store_true")

    args = parser.parse_args()

    main(args)
