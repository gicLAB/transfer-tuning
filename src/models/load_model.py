#!/usr/bin/env python
import os
import pickle
import onnx
import tvm
from tvm import relay


def load_tvm_model(relay_file, relay_params, model_dir: os.PathLike):
    with open(os.path.join(model_dir, relay_file), "r") as fi:
        mod = tvm.ir.load_json(fi.read())
    with open(os.path.join(model_dir, relay_params), "rb") as fi:
        params = relay.load_param_dict(fi.read())
    return mod, params


def import_onnx(model_name: str, file_path: os.PathLike):
    """Load a ONNX model into the TVM ONNX format

    :param model_name:
    :param file_path:
    :returns:

    """

    dirname, _ = os.path.split(file_path)
    config_path = os.path.abspath(dirname)
    config_name = os.path.join(config_path, "models_data.pkl")
    with open(config_name, "rb") as handle:
        (
            input_data,
            model_fnames,
            target_outputs,
            input_names,
            model_arch_names,
        ) = pickle.load(handle)
    input_name_list = input_names[model_name]
    input_shapes = [x.shape for x in input_data[model_name]]
    input_shape_dict = {n: s for n, s in zip(input_name_list, input_shapes)}
    # output_shape = target_outputs[model_name].shape

    model = onnx.load(file_path)
    mod, params = relay.frontend.from_onnx(model, input_shape_dict.copy())

    test_input_data = input_data[model_name]
    target_outputs = target_outputs[model_name]

    test_input_data = {n: data for n, data in zip(input_name_list, test_input_data)}
    return mod, params, test_input_data, target_outputs
