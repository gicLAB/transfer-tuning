#!/usr/bin/env python
import argparse
import os
import pickle
import json
import pathlib
from src.data.transfer_tuning_utils import logfile_splitter


def main(args):

    # get all the models
    models = []
    # for f in os.listdir(args.network_dir):
    #     filename = os.fsdecode(f)
    #     if ".onnx" in filename:
    #         models.append(os.path.join(args.network_dir, filename))
    # make output dir
    op = pathlib.Path(args.output_dir)
    op.mkdir(parents=True, exist_ok=True)

    model_set_info_file = os.path.join(args.network_dir, "models_data.json")
    with open(model_set_info_file) as f:
        model_set_info = json.load(f)

    classes = dict()
    print(f"We have {len(models)} models", models)
    num_wkls = 0
    wkls = []
    for m, model_info in model_set_info.items():
        # if "dcgan" in m:
        #     # not working right now
        #     continue
        #  get the model info (e.g. classes)
        with open(os.path.join(args.network_dir, m + "_wkl_classes.pkl"), "rb") as f:
            wkl_classes = pickle.load(f)[0]
        # print(new_wkl_classes_tuning)
        # raise ValueError()
        num_wkls += len(wkl_classes)
        wkls += list(wkl_classes.keys())
        for k, v in wkl_classes.items():
            if v not in classes:
                classes[v] = []
            classes[v].append(k)

        # split the tuning for said model into separate schedules
        for f in os.listdir(args.log_file_dir):
            filename = os.fsdecode(f)
            if ".best.json" not in filename:
                # needs to be the best tuning
                continue
            if m not in filename:
                # needs to be for that model
                continue
            if "£" not in filename:
                # needs to be a logfile
                continue
            wkl_ids, files = logfile_splitter(
                [os.path.join(args.log_file_dir, filename)], args.output_dir
            )
            print(wkl_ids)
    # save a lookup db of the workload classes
    with open(os.path.join(args.output_dir, "wkl_classes.json"), "w") as f:
        json.dump(classes, f, indent=2)
    print(f"Finished, should have {num_wkls} wkls.")
    print(f"Unique wkls: {len(list(set(wkls)))}")
    wkls_in_our_list = 0
    for k, v in classes.items():
        wkls_in_our_list += len(v)
    print(f"Our dict has {wkls_in_our_list} workloads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Split the logfiles into individual files
        """
    )
    parser.add_argument(
        "--log_file_dir", type=str, required=True, help="Log file to parse"
    )
    parser.add_argument(
        "--network_dir", type=str, required=True, help="ONNX file for network to use"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save split logfiles"
    )
    args = parser.parse_args()
    main(args)
