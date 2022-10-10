#!/usr/bin/env python
import os
import argparse
import json
from json import JSONEncoder
import numpy as np
import os

from src.models.model_sets import model_sets, get_model_set_data
from src.models.export_model import pytorch_onnx_exporter, model_exporter


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def export_json(data: dict, output_file: os.PathLike):
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch models to TVM model format")
    parser.add_argument(
        "--model_set", default="vanilla", type=str, help="Set of models to benchmark"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output dir to save models",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join("models", args.model_set)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_dict = model_sets[args.model_set]

    in_shapes, frameworks = get_model_set_data(model_dict)

    models_data = model_exporter(
        model_dict, in_shapes, frameworks, args.output_dir, verbose=True
    )

    model_data_file = os.path.join(args.output_dir, "models_data.json")
    export_json(models_data, model_data_file)

    print("Exported all models")
