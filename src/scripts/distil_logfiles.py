#!/usr/bin/env python
import argparse
import os
import subprocess


def main(args):
    for f in os.listdir(args.log_file_dir):
        if "json" not in f:
            # it's got to be a JSON file
            continue
        if ".best.json" in f:
            # ignore already distilled file
            continue
        if "£" not in f:
            # probably not a log file - I put £ in mine
            continue
        path = os.path.join(args.log_file_dir, f)
        my_args = [
            "-m",
            "tvm.auto_scheduler.measure_record",
            "--mode",
            "distill",
            "-i",
            path,
        ]
        p = subprocess.run(
            [
                "python3",
                *my_args,
            ]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Use Ansor distillation approach to save the best tuning for each workload
        of each model to its own file
        """
    )
    parser.add_argument(
        "--log_file_dir", type=str, required=True, help="Log file to parse"
    )
    args = parser.parse_args()
    main(args)
