#!/usr/bin/env python
import os
import ast
import re
import json
import numpy as np
from typing import List, Union, Dict
from hashlib import sha256
import tvm
from tvm import auto_scheduler, relay
from tvm.relay.backend.compile_engine import get as get_compile_engine


def args_from_func_string(s):
    return [
        i.strip()
        for i in re.split(
            r"""^\w+\(|\)$|((?:\([^()]*\)|'[^']*'|"[^"]*"|[^'"(),])*)""", s
        )
        if i and i != ","
    ]


def get_shape_tuple(line):
    if "i" in line:
        return None
    stuple = line[line.find("[") : line.find("]")] + "]"

    return list(ast.literal_eval(stuple))


def count_inputs(lines):
    inputs = 0
    for l in lines:
        if "placeholder = PLACEHOLDER" in l:
            inputs += 1
    return inputs


def get_data_padding(line, ih, iw):
    args = args_from_func_string(line)

    padding = [0, 0]
    for a in args:
        if "i2" in a and len(a) > 2:
            a = a.replace("i2", "")
            temp = re.findall(r"\d+", a)
            nums = list(map(int, temp))
            # assert(nums[0] == nums[1] - ih), 'Unexpected assymetric padding'
            padding[0] = nums[0]

        # i3 is split over a few terms, we assume symmetric padding
        if "i3" in a and len(a) > 2:
            a = a.replace("i3", "")
            temp = re.findall(r"\d+", a)
            nums = list(map(int, temp))
            # assert(nums[0] == nums[1] - iw), 'Unexpected assymetric padding'
            padding[1] = nums[0]
            break

    return padding


def get_skip_connection_size(line):
    n, ic_chunk, ih, iw, ic_bn = get_shape_tuple(line)
    shape = (n, (ic_chunk * ic_bn), ih, iw)
    return shape


def get_strides(line):
    args = args_from_func_string(line)
    strides = [1, 1]
    for a in args:
        if "(oh*" in a:
            matches = re.search("\(oh\*(\d)\) \+ kh", a)
            if matches:
                strides[0] = int(matches.groups()[0])
        if "(ow*" in a:
            matches = re.search("\(ow\*(\d)\) \+ kw", a)
            if matches:
                strides[1] = int(matches.groups()[0])

    return strides


def get_pooling_size(in_dim, line):
    args = args_from_func_string(line)
    pool_size = [None, None]
    for a in args:
        if "ax2*" in a:
            temp = re.findall(r"\(ax2\*(\d+)\)", a)
            assert len(temp) == 1
            pool_size[0] = in_dim[0] // int(temp[0])
        elif "ax3*" in a:
            temp = re.findall(r"\(ax3\*(\d+)\)", a)
            assert len(temp) == 1
            pool_size[1] = in_dim[1] // int(temp[0])
    return pool_size


def dense_net_from_wkl(dag, desc):
    params = dict()
    lines = dag.split("\n")

    workload_class = "dense"
    # get data shape
    n, p = get_shape_tuple(lines[0])
    # get weights shape
    units, px = get_shape_tuple(lines[1])

    assert p == px

    assert "T_dense" in lines[2]

    if len(lines) == 4:
        bias = False
    elif len(lines) == 6:
        assert "T_add" in lines[4]
        (units_x,) = get_shape_tuple(lines[3])
        assert units_x == units
        bias = True
        workload_class += "_bias"
    else:
        ...

    for l in lines:
        if "T_add" in l:
            bias = True

    params = {
        "in_features": p,
        "units": units,
        "bias": bias,
    }

    in_shape = [n, p]

    params["class"] = workload_class
    params["in_shape"] = in_shape
    params["n"] = n
    params["output_shape"] = [n, units]

    return (
        {
            "input0": in_shape,
        },
        workload_class,
        params,
    )


def conv2d_net_from_dag(dag, desc):
    params = dict()
    lines = dag.split("\n")
    num_inputs = count_inputs(lines)
    # print(desc, num_inputs)
    # get data shape tuple
    if "DepthwiseConv2d" in dag:
        params["depthwise"] = True
    else:
        params["depthwise"] = False

    params["padding"] = [0, 0]
    use_padding = False
    skip_connection = False

    # check if we are in the three placeholder case:
    three_pl_case = True
    for l in lines[0:3]:
        if "placeholder = PLACEHOLDER" not in l:
            three_pl_case = False
            break

    if three_pl_case:
        skip_connection = True
        skip_shape = get_skip_connection_size(lines[0])

    shape = get_shape_tuple(lines[0 + three_pl_case])
    if len(shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_shape_tuple(lines[0 + three_pl_case])
        ic = ic_chunk * ic_bn
    elif len(shape) == 4:
        n, ic, ih, iw = get_shape_tuple(lines[0 + three_pl_case])
    else:
        raise ValueError("Unexpected shape")

    params["tvm_inshape"] = get_shape_tuple(lines[0 + three_pl_case])

    if "data_pad" in lines[1][0:8] or "PaddedInput" in lines[1]:
        use_padding = True
        params["padding"] = get_data_padding(lines[1 + three_pl_case], ih, iw)

    # get kernel shape tuple
    shape = get_shape_tuple(lines[1 + use_padding + three_pl_case])
    if shape is None and "conv2d_transpose" not in desc:
        raise ValueError("Unknown DAG", f"desc:{desc}", lines)
    elif shape is None and "conv2d_transpose" in desc:
        # set dummy values, as this code isn't used anymore really
        params["in_shape"] = [1, 1]
        params["kernel_shape"] = [1, 1, 1, 1]
        workload_class = "conv2d_transpose"
        return (
            {
                "input0": [1, 1],  # dummy value
            },
            workload_class,
            params,
        )
    elif len(shape) == 6:
        (
            oc_chunk,
            ic_chunk_group,
            kernel_height,
            kernel_width,
            _,
            oc_bn,
        ) = shape
    elif len(shape) == 5:
        skip_connection = True
        if "add_add" in desc:
            skip_init_line = 3
        else:
            skip_init_line = 3
        c = 0
        for l in lines:
            if "placeholder = PLACEHOLDER" in l:
                c += 1
            if c == skip_init_line:
                skip_init_line = l
                break
        skip_shape = get_shape_tuple(skip_init_line)

        (
            oc_chunk,
            ic_chunk_group,
            kernel_height,
            kernel_width,
            _,
            oc_bn,
        ) = skip_shape

    else:
        raise ValueError(f"Unknown shape {shape}, dag is:", lines)
    params["num_inputs"] = num_inputs
    params["in_c"] = ic
    params["num_filters"] = oc_chunk * oc_bn
    params["kdim"] = [kernel_height, kernel_width]

    params["relu"] = False
    params["bias"] = False
    params["relu6"] = False
    params["strides"] = [1, 1]
    for i, l in enumerate(lines):
        if "relu" in l:
            params["relu"] = True
        if "add" in l and not params["bias"]:
            params["bias"] = True
            continue
        if (
            "T_add" in l[0:5]
            and params["bias"]
            and not skip_connection
            and not params["depthwise"]
        ):
            if "compile_engine_const" in lines[i - 1]:
                continue
            skip_connection = True
            try:
                skip_shape = get_skip_connection_size(lines[i - 1])
            except Exception as e:
                print("Line:", lines[i - 1], lines)
                raise e
            continue
        if "conv2d_NCHWc" in l[0:13] or "DepthwiseConv2d" in l[0:15]:
            # getting stride
            params["strides"] = get_strides(l)
        if "6f), 0f)" in l:
            params["relu6"] = True

    in_shape = [n, params["in_c"], ih, iw]

    workload_class = "conv2d"
    if use_padding:
        workload_class += "_padding"
    if params["bias"]:
        workload_class += "_bias"
    if params["relu"]:
        workload_class += "_relu"
    if params["relu6"]:
        workload_class += "_relu6"
    if skip_connection:
        workload_class += "_skip-connection"
    if three_pl_case:
        workload_class += "_three_pl_case"

    params = params.copy()
    params["class"] = workload_class
    params["in_shape"] = in_shape
    params["skip_connection"] = skip_connection
    params["kernel_shape"] = [
        params["num_filters"],
        params["num_filters"],
        kernel_height,
        kernel_width,
    ]
    if use_padding:
        pt, pl = params["padding"]
        pb, pr = pt, pl
    else:
        pt, pl, pb, pr = 0, 0, 0, 0
    sh, sw = params["strides"]
    oh = (ih - kernel_height + pt + pb) // sh + 1
    ow = (iw - kernel_width + pl + pr) // sw + 1

    params["output_shape"] = [n, params["num_filters"], oh, ow]
    return (
        {
            "input0": in_shape,
        },
        workload_class,
        params,
    )


def avgpool_net_from_wkl(dag, desc):
    params = dict()
    lines = dag.split("\n")

    workload_class = "avgpool"
    # get data shape
    n, ic_chunk, ih, iw, ic_bn = get_shape_tuple(lines[0])
    # in_c = ic_chunk * ic_bn
    in_shape = (n, ic_chunk, ih, iw, ic_bn)

    params = dict()
    if "global" in desc:
        params["pool_size"] = [ih, iw]
    else:
        params["pool_size"] = get_pooling_size((ih, iw), lines[1])

    if params["pool_size"] == [None, None]:
        params["pool_size"] = [2, 2]

    params["data_shape"] = in_shape
    params["layout"] = "NCHW" + str(ic_bn)

    params["class"] = workload_class
    params["in_shape"] = in_shape
    # pt, pb = params["padding"][0], params["padding"][0]
    # pl, pr = params["padding"][1], params["padding"][1]
    pt, pb, pl, pr = 0, 0, 0, 0
    sh, sw = 1, 1  # params["strides"]

    oh = (ih - params["pool_size"][0] + pt + pb) // sh + 1
    ow = (iw - params["pool_size"][1] + pl + pr) // sw + 1

    params["output_shape"] = [n, ic_chunk * ic_bn, oh, ow]
    return (
        {
            "data": in_shape,
        },
        workload_class,
        params,
    )


def maxpool_net_from_wkl(dag, desc):
    params = dict()
    lines = dag.split("\n")
    params = dict()
    workload_class = "maxpool"
    # get data shape
    in_shape = get_shape_tuple(lines[0])
    if len(in_shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = in_shape
        params["layout"] = "NCHW" + str(ic_bn)
        ic = ic_chunk * ic_bn
    elif len(in_shape) == 4:
        params["layout"] = "NCHW"
        n, ic, ih, iw = in_shape

    params["data_shape"] = in_shape

    params["padding"] = get_data_padding(lines[1], ih, iw)
    params["strides"] = get_strides(lines[2])
    # get can't seem to recover this info, so we guess
    params["pool_size"] = [2, 2]
    params["class"] = workload_class
    params["in_shape"] = in_shape

    pt, pb = params["padding"][0], params["padding"][0]
    pl, pr = params["padding"][1], params["padding"][1]
    sh, sw = params["strides"]
    oh = (ih - params["pool_size"][0] + pt + pb) // sh + 1
    ow = (iw - params["pool_size"][1] + pl + pr) // sw + 1
    params["output_shape"] = [n, ic, oh, ow]
    return (
        {
            "data": in_shape,
        },
        workload_class,
        params,
    )


def misc_max_min_from_wkl(dag, desc):
    """Don't really know what this dag is"""
    params = dict()
    lines = dag.split("\n")
    params = dict()
    workload_class = "misc_max_min"
    in_shape = get_shape_tuple(lines[0])
    params["data_shape"] = in_shape
    params["class"] = workload_class
    params["in_shape"] = in_shape
    return (
        {
            "data": in_shape,
        },
        workload_class,
        params,
    )


def t_divide_from_wkl(dag, desc):
    """Don't really know what this dag is"""
    params = dict()
    lines = dag.split("\n")
    params = dict()
    workload_class = "t_divide"
    in_shape = get_shape_tuple(lines[0])
    params["data_shape"] = in_shape
    params["class"] = workload_class
    params["in_shape"] = in_shape
    params["output_shape"] = in_shape
    return (
        {
            "data": in_shape,
        },
        workload_class,
        params,
    )


def fused_mean_from_tsk(dag, desc):
    params = dict()
    lines = dag.split("\n")
    params = dict()
    workload_class = "fused_mean"
    in_shape = get_shape_tuple(lines[0])
    params["data_shape"] = in_shape
    params["class"] = workload_class
    params["in_shape"] = in_shape
    # fused mean should probably make 4D go to 2D
    params["output_shape"] = [in_shape[0], in_shape[1]]
    return (
        {
            "data": in_shape,
        },
        workload_class,
        params,
    )


def batch_matmul_from_tsk(dag, desc):
    params = dict()
    lines = dag.split("\n")

    workload_class = "batch_matmul"

    b, i, k = get_shape_tuple(lines[0])

    b2, j, k2 = get_shape_tuple(lines[1])

    assert b == b2
    assert k == k2

    params = dict()
    in_shape = [b, i, k]

    params["class"] = workload_class
    params["in_shape"] = in_shape
    params["other_shape"] = [b, j, k]
    params["output_shape"] = [b, i, j]

    return (
        {
            "input0": in_shape,
        },
        workload_class,
        params,
    )


def get_dag_fn(tgt_dag: str, desc: str):
    if "fused_mean" in desc:
        fn = fused_mean_from_tsk
    elif "conv2d_NCHWc" in tgt_dag:
        fn = conv2d_net_from_dag
    elif "DepthwiseConv2d" in tgt_dag:
        fn = conv2d_net_from_dag
    elif "T_dense" in tgt_dag:
        fn = dense_net_from_wkl
    elif "select((bool)1" in tgt_dag or "avg_pool" in desc:
        fn = avgpool_net_from_wkl
    elif "max=" in tgt_dag:
        fn = maxpool_net_from_wkl
    elif "max" in tgt_dag and "min" in tgt_dag:
        fn = misc_max_min_from_wkl
    elif "T_divide" in tgt_dag:
        fn = t_divide_from_wkl
    elif "fused_nn.batch_matmul" == desc:
        fn = batch_matmul_from_tsk
    else:
        fn = None
    return fn


def get_workload_info(
    tasks: List[auto_scheduler.SearchTask],
    verbose: bool = False,
    skip_error: bool = False,
) -> Union[Dict, Dict]:
    """Given a list of Ansor tasks, return their classes and full workload ids

    :param tasks:
    :param verbose:
    :param skip_error:
    :returns:

    """

    workload_classes = dict()
    full_wkl_ids = dict()  # deal with TVMs new verbose format
    for i, tsk in enumerate(tasks):
        tgt_wkl_id = tsk.workload_key
        tgt_dag = str(tsk.compute_dag)
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()
        wkl_desc = tsk.desc

        fn = get_dag_fn(tgt_dag, wkl_desc)
        if fn is None:
            if verbose:
                print("Unknown dag")
                print(tgt_dag)
            if not skip_error:
                raise ValueError("Unknown Dag", tgt_dag)
            continue
        if verbose:
            print(f"\n\nChecking workload{tgt_wkl_id}"), print(tgt_dag)
        net_in_shape_dict, workload_class, params = fn(tgt_dag, wkl_desc)

        workload_classes[tgt_wkl_id_sha] = wkl_desc  # workload_class
        full_wkl_ids[tgt_wkl_id_sha] = tgt_wkl_id
    return workload_classes, full_wkl_ids


def classes_to_node_name(classes: List[str]) -> List[str]:
    """Generate names for each workoad, from the class name

    :param classes:
    :returns:

    """
    counts = Counter(classes)
    curr_count = dict()
    for i, c in enumerate(classes):
        if counts[c] > 1:
            if c not in curr_count:
                classes[i] = c  # + "_1" # zero'th occurence
                curr_count[c] = 1
            else:
                classes[i] = c + "_" + str(curr_count[c])
                curr_count[c] += 1

    return classes


def get_workload_info_and_params(
    tasks: List[auto_scheduler.SearchTask],
    verbose=False,
    skip_error=False,
    readable_names=False,
):
    workload_classes = dict()
    full_wkl_ids = dict()  # deal with TVMs new verbose format
    all_params = dict()
    human_names = dict()
    for i, tsk in enumerate(tasks):
        tgt_wkl_id = tsk.workload_key
        tgt_dag = str(tsk.compute_dag)
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()
        tsk_desc = tsk.desc
        human_name = tsk_desc + "_" + tgt_wkl_id_sha[0:6]
        human_names[tgt_wkl_id_sha] = human_name
        fn = get_dag_fn(tgt_dag, tsk_desc)
        if fn is None:
            if verbose:
                print("Unknown dag")
                print(tgt_dag)
            if not skip_error:
                raise ValueError("Unknown Dag", tgt_dag, tsk_desc)
            continue
        if verbose:
            print(f"\n\nChecking workload{tgt_wkl_id}"), print(tgt_dag)
        net_in_shape_dict, workload_class, params = fn(tgt_dag, tsk_desc)
        workload_class = tsk_desc
        wkl_id = tgt_wkl_id[2:34]
        params["wkl_id"] = wkl_id
        params["wkl_class"] = tsk_desc  # workload_class
        if "onv2d" in tgt_dag:
            params["size"] = np.prod(params["in_shape"]) * np.prod(
                params["kernel_shape"]
            )
        elif "dense" in tgt_dag:
            params["size"] = params["units"] * np.prod(params["in_shape"])
        else:
            params["size"] = np.prod(params["in_shape"])
        all_params[tgt_wkl_id_sha] = params
        workload_classes[tgt_wkl_id_sha] = workload_class
        full_wkl_ids[tgt_wkl_id_sha] = tgt_wkl_id
    if readable_names:
        return workload_classes, full_wkl_ids, all_params, human_names
    else:
        return workload_classes, full_wkl_ids, all_params


def _NCHWc_to_NCHW(shape):
    shape = [shape[0], shape[1] * shape[-1], shape[2], shape[3]]
    return shape


def _6d_to_4d(shape):
    oc_chunk, ic_chunk_group, kh, kw, ic_bn, oc_bn = shape
    oc = oc_chunk * oc_bn
    ic = ic_chunk_group * ic_bn
    shape = [oc, ic, kh, kw]
    return shape


def _get_graph_tensor_shapes(graph_dump_node, graph_dump_nodes, tensor_num):
    if len(graph_dump_node["inputs"]) < tensor_num + 1:
        print(f"No tensors for {graph_dump_node['op']}")
        print(graph_dump_node["inputs"])
        return []

    tensor_name = graph_dump_node["inputs"][tensor_num]
    for n in graph_dump_nodes:
        if n["name"] == tensor_name:
            tensor_shape = n["shape"]
            if "NCHWc_add_clip_19" in tensor_name:
                print(
                    "hey _get_graph_tensor_shapes!",
                    tensor_name,
                    "input shape",
                    tensor_shape,
                )
            if len(tensor_shape) == 5:
                tensor_shape = _NCHWc_to_NCHW(tensor_shape)
            if "NCHWc_add_clip_19" in tensor_name:
                print("now it's hey!", tensor_name, "input shape", tensor_shape)
                print()
    return tensor_shape


def _update_graph_json(graph_json):
    for i, n in enumerate(graph_json["nodes"]):
        try:
            x = n["op"]
        except:
            print("Error with", n)
        if n["op"] == "null":
            continue
        elif n["op"] == "param":
            continue
        elif "fused_layout_transform" in n["name"]:
            continue
        elif "input" in n["name"]:
            continue
        elif "__nop" in n["name"]:
            continue
        elif re.match(r"p[0-9]+", n["name"]):
            continue
        elif "fused_mean" in n["name"] or "nn_batch_flatten" in n["name"]:
            input_shape = _get_graph_tensor_shapes(n, graph_json["nodes"], 0)

            graph_json["nodes"][i]["input_shape"] = input_shape
            out_shape = n["shape"]
            if len(out_shape) == 5:
                out_shape = _NCHWc_to_NCHW(out_shape)
            graph_json["nodes"][i]["output_shape"] = out_shape
        else:
            input_shape = _get_graph_tensor_shapes(n, graph_json["nodes"], 0)

            graph_json["nodes"][i]["input_shape"] = input_shape
            out_shape = n["shape"]
            if len(out_shape) == 5:
                out_shape = _NCHWc_to_NCHW(out_shape)
            graph_json["nodes"][i]["output_shape"] = out_shape

            graph_json["nodes"][i]["param_shape"] = _get_graph_tensor_shapes(
                n, graph_json["nodes"], 1
            )
    return graph_json


def _get_graph_dump_data(graph_dump_nodes, drop_contrib_NCHWc=False):
    trace_output_shapes = dict()
    trace_in_shapes = dict()
    trace_ops = dict()
    trace_ksize = dict()
    for n in graph_dump_nodes:
        if n["op"] == "null":
            continue
        elif "fused_layout_transform" in n["name"]:
            continue
        elif "fused_nn_batch_flatten" in n["name"]:
            continue
        elif "input" in n["name"]:
            continue
        elif re.match(r"p[0-9]+", n["name"]):
            continue
        elif n["op"] == "param":
            continue
        else:
            nme = n["name"]
            try:
                input_shape = n["input_shape"]
            except Exception as e:
                print("error with node", n["name"], n)
                raise e
            output_shape = n["output_shape"]

            if "NCHWc_add_clip_19" in n["name"]:
                print("hey!", nme, "input shape", input_shape)
            if len(input_shape) == 5:
                raise ValueError("This should already be 4D")
                input_shape = _NCHWc_to_NCHW(input_shape)
            trace_output_shapes[nme] = output_shape

            trace_in_shapes[nme] = input_shape

            if "conv2d" in nme:
                trace_ksize[nme] = [n["param_shape"][2], n["param_shape"][3]]
            else:
                trace_ksize[nme] = None
            op_name = nme
            # get the generic op name
            if op_name.split("_")[-1].isdigit():
                op_name = "_".join(op_name.split("_")[:-1])
            while op_name[-1].isdigit():
                op_name = op_name[:-1]

            if drop_contrib_NCHWc:
                op_name = op_name.replace("bias", "add")
                op_name = op_name.replace("add_add", "add")
                if "bias" in op_name:
                    raise ValueError("Should have passed here")

            trace_ops[nme] = op_name

    return trace_output_shapes, trace_ops, trace_in_shapes, trace_ksize


def _get_auto_sched_info(tasks, task_weights, all_params, drop_contrib_NCHWc=False):
    shas = dict()
    sha_weights = dict()
    sha_dags = dict()
    for tsk, weight in zip(tasks, task_weights):
        desc = tsk.desc
        tgt_wkl_id = tsk.workload_key
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()

        shas[tgt_wkl_id_sha] = desc
        sha_weights[tgt_wkl_id_sha] = weight
        sha_dags[tgt_wkl_id_sha] = tsk.compute_dag

    wkl_input_shapes = dict()
    wkl_out_shapes = dict()
    wkl_ops = dict()
    wkl_weights = dict()
    wkl_params = dict()
    wkl_ids = dict()
    wkl_dags = dict()
    for wkl_id, desc in shas.items():
        descc = desc + "_" + wkl_id[0:5]
        wkl_ids[descc] = wkl_id
        wkl_dags[descc] = sha_dags[wkl_id]
        in_shape = all_params[wkl_id]["in_shape"]
        if len(in_shape) == 5:
            in_shape = _NCHWc_to_NCHW(in_shape)

        wkl_input_shapes[descc] = in_shape

        try:
            out_shape = all_params[wkl_id]["output_shape"]
        except Exception as e:
            print("No output shape for workload:", descc)
            print(all_params[wkl_id])
            raise e
        wkl_out_shapes[descc] = out_shape

        wkl_weights[descc] = sha_weights[wkl_id]

        wkl_ops[descc] = desc.replace(".", "_")
        if drop_contrib_NCHWc:
            wkl_ops[descc] = wkl_ops[descc].replace(
                "fused_nn_contrib_conv2d_NCHWc_", "fused_nn_conv2d_nn_"
            )
        wkl_params[descc] = all_params[wkl_id]

    return (
        wkl_out_shapes,
        wkl_input_shapes,
        wkl_ops,
        wkl_weights,
        wkl_params,
        wkl_ids,
        wkl_dags,
    )


# def get_tracing_names_2(
#     profile_dir,
#     wkl_full_params,
#     tasks: List[auto_scheduler.SearchTask],
#     task_weights: List[int],
# ):
#     # this step could be performed with zero overhead
#     # if we had better access to TVM's various internal
#     # naming schemes
#     # get profiling names
#     prof_file = os.path.join(profile_dir, "_tvmdbg_graph_dump.json")
#     with open(prof_file) as f:
#         graph_json = json.load(f)
#     graph_json = _update_graph_json(graph_json)

#     trace_output_shapes, trace_ops, trace_in_shapes, trace_ksize = _get_graph_dump_data(
#         graph_json["nodes"]
#     )

#     (
#         wkl_out_shapes,
#         wkl_input_shapes,
#         wkl_ops,
#         wkl_weights,
#         wkl_params,
#         wkl_ids,
#         wkl_dags,
#     ) = _get_auto_sched_info(tasks, task_weights, wkl_full_params)

#     # match the auto_sched names to profiling names
#     potentials = dict()
#     orig_trace_names = list(trace_output_shapes.keys())
#     trace_names_orig_len = len(orig_trace_names)
#     used_trace_names = (
#         []
#     )  # keep track of what trace names we've used, alert to doublers
#     trace_name_lookup = dict()  # track what workload we allocate a trace_name to
#     debug = False
#     for i, (wkl_name, wkl_op) in enumerate(wkl_ops.items()):
#         if debug:
#             print("Checking", i, wkl_name)

#         potentials[wkl_name] = []
#         wkl_out_shape = wkl_out_shapes[wkl_name]
#         for trce_name, trce_out_shape in trace_output_shapes.items():

#             if wkl_op != trace_ops[trce_name]:
#                 continue
#             if "dense" in trce_name:
#                 print("checking", trce_name)
#             if "conv2d" in trce_name:
#                 if wkl_params[wkl_name]["kdim"] != trace_ksize[trce_name]:
#                     continue
#             if "max_pool" in trce_name:
#                 wkl_out_shape = trce_out_shape  # we can't easily get the correct out shape from the task
#             if "avg_pool" in trce_name:
#                 wkl_out_shape = trce_out_shape  # we can't easily get the correct out shape from the task
#             if wkl_input_shapes[wkl_name] != trace_in_shapes[trce_name]:
#                 if debug:
#                     print(
#                         f"Wrong inshape for `{trce_name}` vs `{wkl_name}`: `{trace_in_shapes[trce_name]}` vs `{wkl_input_shapes[wkl_name]}`"
#                     )
#                 continue
#             else:
#                 if debug:
#                     print(f"Found right inshape!")

#             if wkl_out_shape != trce_out_shape:
#                 if debug:
#                     print(
#                         f"Wrong outshape for `{trce_name}` vs `{wkl_name}`: `{trce_out_shape}` vs `{wkl_out_shape}`"
#                     )
#                 continue
#             else:
#                 if debug:
#                     print(f"Found right outshape!")

#             potentials[wkl_name].append(trce_name)
#             if trce_name in used_trace_names:
#                 print(trce_name, wkl_name)
#                 print(trace_name_lookup[trce_name])
#                 raise ValueError(
#                     "Ambiguity in workloads, cannot use the same trace name twice"
#                 )
#             used_trace_names.append(trce_name)
#             trace_name_lookup[trce_name] = wkl_name

#     profiling_names = dict()
#     error = False
#     total_weight = 0
#     used_trace_names = []
#     # verify that things are probably fine
#     # update the names to be associated with workload ids
#     for wkl_name, trace_names in potentials.items():
#         wkl_weight = wkl_weights[wkl_name]
#         total_weight += wkl_weight
#         wkl_id = wkl_ids[wkl_name]
#         used_trace_names += trace_names
#         if len(trace_names) != wkl_weight:
#             print(
#                 "Unexpected number of workloads",
#                 wkl_name,
#                 f"Wkl weight: {wkl_weight}",
#                 f"len(trace_names): {len(trace_names)}",
#                 trace_names,
#                 wkl_out_shapes[wkl_name],
#                 wkl_input_shapes[wkl_name],
#                 trace_in_shapes[trce_name],
#                 wkl_params[wkl_name],
#             )
#             error = True
#         profiling_names[wkl_id] = trace_names
#     if error:
#         print(potentials)
#         print("Remaining trace_names:", trace_names)
#         print("Excepted weights:", wkl_weights)
#         print("Used trace names:", used_trace_names)
#         print(
#             "Unused trace names",
#             [x for x in orig_trace_names if x not in used_trace_names],
#         )
#         print(trace_names_orig_len, total_weight)
#         print("Orig trace names:", list(trace_output_shapes.keys()))
#         raise ValueError("Unexpected number of workloads")

#     return profiling_names


def get_tracing_names(mod, params, test_inputs, tasks, task_weights):
    # this step could be performed with zero overhead
    # if we had better access to TVM's various internal
    # naming schemes

    # get profiling names
    profile_dir = "/tmp/"
    from tvm.contrib.debugger.debug_executor import GraphModuleDebug

    target = "llvm"
    ctx = tvm.cpu(0)
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        lib = relay.build(mod, target=target, params=params)
    module = GraphModuleDebug(
        lib["debug_create"]("default", ctx),
        [ctx],
        lib.graph_json,
        dump_root=profile_dir,
    )
    dtype = "float32"
    for input_name, test_input in test_inputs.items():
        data_tvm = tvm.nd.array(test_input.astype(dtype))
        module.set_input(input_name, data_tvm)
    module.run()

    _, _, wkl_full_params = get_workload_info_and_params(tasks)

    profile_dir = os.path.join(profile_dir, "_tvmdbg_device_CPU_0")
    return get_tracing_names_2(
        profile_dir,
        wkl_full_params,
        tasks,
        task_weights,
    )


def get_tracing_names_2(
    profile_dir,
    wkl_full_params,
    tasks: List[auto_scheduler.SearchTask],
    task_weights: List[int],
    drop_contrib_NCHWc=False,
):
    # this step could be performed with zero overhead
    # if we had better access to TVM's various internal
    # naming schemes
    # get profiling names
    prof_file = os.path.join(profile_dir, "_tvmdbg_graph_dump.json")
    with open(prof_file) as f:
        graph_json = json.load(f)
    graph_json = _update_graph_json(graph_json)

    trace_output_shapes, trace_ops, trace_in_shapes, trace_ksize = _get_graph_dump_data(
        graph_json["nodes"], drop_contrib_NCHWc=drop_contrib_NCHWc
    )

    (
        wkl_out_shapes,
        wkl_input_shapes,
        wkl_ops,
        wkl_weights,
        wkl_params,
        wkl_ids,
        wkl_dags,
    ) = _get_auto_sched_info(tasks, task_weights, wkl_full_params, drop_contrib_NCHWc)

    # match the auto_sched names to profiling names
    potentials = dict()
    orig_trace_names = list(trace_output_shapes.keys())
    trace_names_orig_len = len(orig_trace_names)
    used_trace_names = (
        []
    )  # keep track of what trace names we've used, alert to doublers
    trace_name_lookup = dict()  # track what workload we allocate a trace_name to
    debug = False
    for i, (wkl_name, wkl_op) in enumerate(wkl_ops.items()):

        # if wkl_name == "fused_nn.contrib_conv2d_NCHWc_add_nn.relu_1056b":
        #     debug = True
        # else:
        #     continue

        if debug:
            print("Checking", i, wkl_name)
        potentials[wkl_name] = []
        wkl_out_shape = wkl_out_shapes[wkl_name]
        for trce_name, trce_out_shape in trace_output_shapes.items():

            if wkl_op != trace_ops[trce_name]:
                continue
            if "dense" in trce_name:
                print("checking", trce_name)
            if "conv2d" in trce_name:

                if wkl_params[wkl_name]["kdim"] != trace_ksize[trce_name]:
                    if debug:
                        print(f"Wrong kdim for `{trce_name}` vs `{wkl_name}``")
                    continue
            if "max_pool" in trce_name:
                wkl_out_shape = trce_out_shape  # we can't easily get the correct out shape from the task
            if "avg_pool" in trce_name:
                wkl_out_shape = trce_out_shape  # we can't easily get the correct out shape from the task
            if wkl_input_shapes[wkl_name] != trace_in_shapes[trce_name]:
                if debug:
                    print(
                        f"Wrong inshape for `{trce_name}` vs `{wkl_name}`: `{trace_in_shapes[trce_name]}` vs `{wkl_input_shapes[wkl_name]}`"
                    )
                continue
            else:
                if debug:
                    print(f"Found right inshape!")

            if wkl_out_shape != trce_out_shape:
                if debug:
                    print(
                        f"Wrong outshape for `{trce_name}` vs `{wkl_name}`: `{trce_out_shape}` vs `{wkl_out_shape}`"
                    )
                continue
            else:
                if debug:
                    print(f"Found right outshape!")

            if debug:
                print(f"Adding something for {wkl_name}")
            potentials[wkl_name].append(trce_name)
            if trce_name in used_trace_names:
                print(trce_name, wkl_name)
                print(trace_name_lookup[trce_name])
                raise ValueError(
                    "Ambiguity in workloads, cannot use the same trace name twice"
                )
            used_trace_names.append(trce_name)
            trace_name_lookup[trce_name] = wkl_name

    profiling_names = dict()
    error = False
    total_weight = 0
    used_trace_names = []
    # verify that things are probably fine
    # update the names to be associated with workload ids
    for wkl_name, trace_names in potentials.items():
        wkl_weight = wkl_weights[wkl_name]
        total_weight += wkl_weight
        wkl_id = wkl_ids[wkl_name]
        used_trace_names += trace_names
        if len(trace_names) != wkl_weight:
            print(
                "Unexpected number of workloads",
                wkl_name,
                f"Wkl weight: {wkl_weight}",
                f"len(trace_names): {len(trace_names)}",
                trace_names,
                wkl_out_shapes[wkl_name],
                wkl_input_shapes[wkl_name],
                trace_in_shapes[trce_name],
                wkl_params[wkl_name],
            )
            error = True
        profiling_names[wkl_id] = trace_names
    if error:
        print("Error!\n\n")
        print("potentials:", potentials)
        print("Remaining trace_names:", trace_names)
        print("Excepted weights:", wkl_weights)
        print("Used trace names:", used_trace_names)
        print(
            "Unused trace names",
            [x for x in orig_trace_names if x not in used_trace_names],
        )
        print()
        print(trace_names_orig_len, total_weight)
        # print("Orig trace names:", list(trace_output_shapes.keys()))
        # print()
        print()
        print(wkl_ops.keys())
        print()
        print(trace_ops.keys())
        print()
        print("wkl_ops = ", wkl_ops)
        print()
        print("trace_ops = ", trace_ops)
        raise ValueError("Unexpected number of workloads")

    return profiling_names


def get_tensor_shape(tensor_list):
    in_shapes = []
    for t in tensor_list:
        shape = list(t.shape)
        if len(shape) == 5:
            shape = _NCHWc_to_NCHW(shape)
        elif len(shape) == 6:
            shape = _6d_to_4d(shape)
        else:
            ...

        in_shapes.append(shape)
    return in_shapes


def get_cache_funcs():
    ceng = get_compile_engine()
    cache_items = ceng.items()
    cache_funcs = dict()

    for k, v in cache_items:
        if v.cached_func is None:
            continue
        func_name = v.cached_func.func_name
        if "fused_expand_dims" in func_name:
            continue
        elif "fused_layout_transform" in func_name:
            continue
        cache_funcs[func_name] = dict()
        cache_funcs[func_name]["use_count"] = v.use_count
        # cache_funcs[func_name]["source_func"] = k.source_func.astext()
        # print(func_name)
        cache_funcs[func_name]["input_shapes"] = get_tensor_shape(v.cached_func.inputs)
        cache_funcs[func_name]["output_shapes"] = get_tensor_shape(
            v.cached_func.outputs
        )

        # get the generic op name
        if func_name.split("_")[-1].isdigit():
            op_name = "_".join(func_name.split("_")[:-1])
        while op_name[-1].isdigit():
            op_name = op_name[:-1]
        cache_funcs[func_name]["op_name"] = op_name
    return cache_funcs


def _get_auto_sched_info_v2(tasks, task_weights, all_params):
    shas = dict()
    sha_weights = dict()
    sha_dags = dict()
    for tsk, weight in zip(tasks, task_weights):
        desc = tsk.desc
        tgt_wkl_id = tsk.workload_key
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()

        shas[tgt_wkl_id_sha] = desc
        sha_weights[tgt_wkl_id_sha] = weight
        sha_dags[tgt_wkl_id_sha] = tsk.compute_dag

    wkl_funcs = dict()
    for wkl_id, desc in shas.items():
        descc = desc + "_" + wkl_id[0:5]
        wkl_funcs[wkl_id] = dict()
        wkl_funcs[wkl_id]["func_name"] = descc
        wkl_funcs[wkl_id]["op_name"] = desc.replace(".", "_")
        wkl_funcs[wkl_id]["params"] = all_params[wkl_id]
        wkl_funcs[wkl_id]["use_count"] = sha_weights[wkl_id]

        # get input shapes
        in_shape = all_params[wkl_id]["in_shape"]
        if len(in_shape) == 5:
            in_shape = _NCHWc_to_NCHW(in_shape)

        wkl_funcs[wkl_id]["input_shapes"] = in_shape

        # get output shapes
        try:
            out_shape = all_params[wkl_id]["output_shape"]
        except Exception as e:
            print("No output shape for workload:", descc)
            print(all_params[wkl_id])
            raise e
        wkl_funcs[wkl_id]["output_shapes"] = out_shape

    return wkl_funcs


def get_cache_names(tasks, task_weights, wkl_full_params):
    cache_funcs = get_cache_funcs()

    wkl_funcs = _get_auto_sched_info_v2(tasks, task_weights, wkl_full_params)

    wkl_id_cache_lookup = dict()
    for wkl_id, wkl_data in wkl_funcs.items():
        wkl_id_cache_lookup[wkl_id] = []

        for cache_fname, cdata in cache_funcs.items():
            # check if the ops are the same
            if wkl_data["op_name"] != cdata["op_name"]:
                continue

            # maxpooling currently doesn't work

            # check if the output shapes are the same
            if wkl_data["output_shapes"] not in cdata["output_shapes"]:
                continue

            # check if the input shapes are the same
            if wkl_data["input_shapes"] not in cdata["input_shapes"]:
                continue

            wkl_id_cache_lookup[wkl_id].append(cache_fname)
    return wkl_id_cache_lookup
