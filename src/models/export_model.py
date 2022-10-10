#!/usr/bin/env python
import os
import pickle
import numpy as np

# import tensorflow as tf
import torch
import tempfile
import onnx
from typing import Dict
import tvm
from tvm import relay, runtime


def pytorch_onnx_exporter(nets, in_shapes, output_dir, verbose=True):
    # get target outputs from single layer networks
    # export to ONNX format
    input_data, input_names, target_outputs = dict(), dict(), dict()
    model_archs, model_fnames = dict(), dict()

    for i, (name, model) in enumerate(nets.items()):
        if verbose:
            print(f"{name} ({i+1}/{len(nets.keys())})")
        in_shape_list = in_shapes[name]

        test_input_datas = tuple(torch.randn(i) for i in in_shape_list)
        # test_input = Variable(torch.rand(in_shape))
        model.eval()

        if "stylegan2" in name:
            outs = model(*test_input_datas, c=None)
        else:
            outs = model(*test_input_datas)
        target_outputs[name] = outs[0].detach().numpy()
        input_data[name] = [x.numpy() for x in test_input_datas]
        input_names[name] = [f"input{i}" for i in range(len(test_input_datas))]

        # export to onnx
        model_fnames[name] = name + ".onnx"
        model_archs[name] = name
        save_name = os.path.join(output_dir, name + ".onnx")
        torch.onnx.export(
            model, test_input_datas, save_name, input_names=input_names[name]
        )

    # save all information to a pickle file
    models_data = (input_data, model_fnames, target_outputs, input_names, model_archs)
    save_name = os.path.join(output_dir, "models_data.pkl")
    with open(save_name, "wb") as f:
        pickle.dump(models_data, f)

    return models_data


def save_tvm_model(
    mod,
    params,
    output_dir: os.PathLike,
    relay_file="model.json",
    relay_params="model.params",
):
    with open(os.path.join(output_dir, relay_file), "w") as fo:
        fo.write(tvm.ir.save_json(mod))
    with open(os.path.join(output_dir, relay_params), "wb") as fo:
        fo.write(runtime.save_param_dict(params))
    return


def pytorch_exporter(model_name, model_func, in_shape_list, model_data, output_dir):
    model = model_func()  # fetch the model
    # for (i, t) in in_shape_list:
    #     print(i, t)

    torch_dtypes = {
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    test_input_datas = tuple(
        torch.randn(*s, dtype=torch_dtypes[t]) for (s, t) in in_shape_list
    )
    model.eval()

    outs = model(*test_input_datas)
    model_data["target_outputs"] = outs[0].detach().numpy()
    model_data["test_input_data"] = {
        f"input{i}": (x.numpy(), t)
        for i, (x, (_, t)) in enumerate(zip(test_input_datas, in_shape_list))
    }
    model_data["input_shapes"] = {
        n: d.shape for n, (d, _) in model_data["test_input_data"].items()
    }
    input_names = list(model_data["test_input_data"].keys())

    if "stylegan2" in model_name:
        t = (torch.randn((1, 512), dtype=torch.float32).cpu(),)
        model = torch.jit.trace(model, t).eval()

    # exporting to ONNX seems to be more resilient
    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_file_h:
        onnx_file = onnx_file_h.name
        if "stylegan2" in model_name:
            # torch.onnx.export(
            #     model,
            #     *test_input_datas,
            #     onnx_file,
            #     input_names=input_names,
            #     opset_version=10,
            #     export_params=True,
            # )
            mod, params = relay.frontend.from_pytorch(model, [("input0", [1, 512])])
        else:
            torch.onnx.export(
                model, test_input_datas, onnx_file, input_names=input_names
            )
            onnx_model = onnx.load(onnx_file)

            # export to TVM
            mod, params = relay.frontend.from_onnx(
                onnx_model, model_data["input_shapes"]
            )

        relay_file = model_name + ".json"
        relay_params = model_name + ".params"
        save_tvm_model(mod, params, output_dir, relay_file, relay_params)
        model_data["fnames"] = (relay_file, relay_params)
    return model_data


def hf_transformers_exporter(
    model_name, model_func, in_shape_list, model_data, output_dir
):
    batch_size, seq_len = in_shape_list[0]
    model, test_inputs, target_outputs = model_func(
        batch_size, seq_len
    )  # fetch the model

    model_data["target_outputs"] = target_outputs
    model_data["test_input_data"] = test_inputs
    model_data["input_shapes"] = {
        n: d.shape for n, (d, _) in model_data["test_input_data"].items()
    }

    # export to TVM
    mod, params = relay.frontend.from_tensorflow(
        model, shape=model_data["input_shapes"]
    )

    relay_file = model_name + ".json"
    relay_params = model_name + ".params"
    save_tvm_model(mod, params, output_dir, relay_file, relay_params)
    model_data["fnames"] = (relay_file, relay_params)
    return model_data


# def keras_exporter(model_name, model_func, in_shape_list, model_data, output_dir):
#     tf.keras.backend.set_image_data_format("channels_first")
#     model = model_func()  # fetch the model
#     test_input_datas = tuple(np.random.rand(*s) for s in in_shape_list)

#     outs = model.predict((*test_input_datas))
#     print(outs)
#     model_data["target_outputs"] = outs
#     model_data["test_input_data"] = {
#         f"input{i}": x for i, x in enumerate(test_input_datas)
#     }
#     model_data["input_shapes"] = {
#         n: d.shape for n, d in model_data["test_input_data"].items()
#     }
#     input_names = list(model_data["test_input_data"].keys())

#     mod, params = relay.frontend.from_keras(model, model_data["input_shapes"])

#     relay_file = model_name + ".json"
#     relay_params = model_name + ".params"
#     save_tvm_model(mod, params, output_dir, relay_file, relay_params)
#     model_data["fnames"] = (relay_file, relay_params)

#     return model_data


def model_exporter(
    nets, in_shapes, frameworks: Dict[str, str], output_dir: os.PathLike, verbose: True
):
    models_data = dict()
    for i, (name, model_func) in enumerate(nets.items()):
        if verbose:
            print(f"{name} ({i+1}/{len(nets.keys())})")
        models_data[name] = dict()
        in_shape_list = in_shapes[name]

        if frameworks[name] == "pytorch":
            models_data[name] = pytorch_exporter(
                name, model_func, in_shape_list, models_data[name], output_dir
            )
        elif frameworks[name] == "keras":
            models_data[name] = keras_exporter(
                name, model_func, in_shape_list, models_data[name], output_dir
            )
        elif frameworks[name] == "hf-transformers":
            models_data[name] = hf_transformers_exporter(
                name, model_func, in_shape_list, models_data[name], output_dir
            )
        else:
            raise ValueError("Unknown model framework:", frameworks[name])

    return models_data
