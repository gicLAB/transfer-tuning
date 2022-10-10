#!/usr/bin/env python
import os
import sys
import json
import pathlib
import logging
from tvm import auto_scheduler
from src.models.load_model import load_tvm_model
from src.data.workload_utils import get_workload_info_and_params


def get_device(
    device_name: str,
    device_json: os.PathLike = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "device_info.json"
    ),
):
    with open(device_json) as f:
        data = json.load(f)
    if device_name in data.keys():
        return data[device_name]
    else:
        raise ValueError(
            f"Device {device_name} not found in device info file {device_json}"
        )


def setup_tt_logger(logfile):
    open(logfile, "w").close()
    logging.basicConfig(
        filename=logfile,
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("transfer-tuning")
    # output to stdout too
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# def get_model_info(
#     network_name: str,
#     network_file: os.PathLike,
#     device_info,
# ):
#     mod, params, test_inputs, target_outputs = import_onnx(network_name, network_file)
#     tasks, task_weights = auto_scheduler.extract_tasks(
#         mod["main"],
#         params,
#         device_info["target"],
#         device_info["host"],
#     )
#     # wkl_classes, full_wkl_ids = get_workload_info(tasks)
#     (
#         wkl_classes,
#         full_wkl_ids,
#         wkl_full_params,
#         wkl_names,
#     ) = get_workload_info_and_params(tasks, readable_names=True)

#     # # get wkl_names
#     # with tvm.transform.PassContext(
#     #     opt_level=3, config={"relay.backend.use_auto_scheduler": True}
#     # ):
#     #     lib = relay.build(
#     #         mod,
#     #         target=device_info["target"],
#     #         target_host=device_info["host"],
#     #         params=params,
#     #     )
#     # nodes = ast.literal_eval(lib.graph_json)["nodes"]
#     # wkl_names = []
#     # for n in nodes:
#     #     if n["op"] == "null":
#     #         continue
#     #     elif "fused_layout_transform" in n["name"]:
#     #         continue
#     #     else:
#     #         wkl_names.append(n["name"])

#     # wkl_names = list(wkl_classes.values())
#     # wkl_ids = list(wkl_classes.keys())
#     # wkl_names = [x.replace(".", "_") for x in wkl_names]
#     # wkl_names = wkl_names[::-1]  # reverse list
#     # wkl_ids = wkl_ids[::-1]  # reverse list
#     # wkl_names = {
#     #     wkl_id: name for wkl_id, name in zip(wkl_ids, classes_to_node_name(wkl_names))
#     # }
#     # _ = get_tracing_names(mod, params, test_inputs, tasks, task_weights)
#     # print(len(wkl_names))
#     # print(wkl_names)
#     # wkl_names = {wkl_id: name for wkl_id, name in zip(wkl_ids, wkl_names)}

#     return (
#         mod,
#         params,
#         test_inputs,
#         target_outputs,
#         tasks,
#         wkl_classes,
#         full_wkl_ids,
#         task_weights,
#         wkl_names,
#         wkl_full_params,
#     )


def get_model_info(
    network_name: str,
    network_path: os.PathLike,
    device_info,
):
    model_set_info_file = os.path.join(network_path, "models_data.json")
    with open(model_set_info_file) as f:
        model_info = json.load(f)[network_name]

    relay_file, relay_params = model_info["fnames"]
    mod, params = load_tvm_model(relay_file, relay_params, network_path)

    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        device_info["target"],
        device_info["host"],
    )

    (
        wkl_classes,
        full_wkl_ids,
        wkl_full_params,
        wkl_names,
    ) = get_workload_info_and_params(tasks, readable_names=True)

    return (
        mod,
        params,
        model_info["test_input_data"],
        model_info["target_outputs"],
        tasks,
        wkl_classes,
        full_wkl_ids,
        task_weights,
        wkl_names,
        wkl_full_params,
    )
