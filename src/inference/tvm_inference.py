#!/usr/bin/env python
import numpy as np
import tvm
import pickle
import os
import tempfile
import subprocess
import pathlib
import time
from typing import Dict, Optional
from tvm import relay, auto_scheduler, rpc
from tvm.contrib import utils
from tvm.contrib.debugger.debug_executor import GraphModuleDebug


def get_target(device_info):
    if "remote" == device_info["host_type"]:
        runner = auto_scheduler.RPCRunner(
            device_info["key"],
            device_info["address"],
            device_info["port"],
            repeat=3,
            timeout=50,
        )
        target = tvm.target.Target(device_info["target"])
        target_host = tvm.target.Target(device_info["host"])
    elif "local" == device_info["host_type"]:
        runner = auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True)
        target = device_info["target"]
        target_host = device_info["host"]
    else:
        raise ValueError(f"Unknwon host type: {device_info['host_type']}")
    return runner, target, target_host


def evaluate_untuned_ansor(
    mod: tvm.ir.module.IRModule,
    params: Dict,
    device_info,
    test_input_data,
    target_outputs,
    use_ndk=False,
    dtype="float32",
    hardware_params=None,
    profile: bool = False,
    profile_dir: os.PathLike = "/tmp/tvmdbg",
    runs: Optional[int] = 10,
):
    runner, target, target_host = get_target(device_info)

    # Compile the whole network, with no auto-schedule file
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    ):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    print("Compiled model")
    return _evaluate_model(
        lib,
        device_info,
        test_input_data,
        target_outputs,
        dtype=dtype,
        use_ndk=use_ndk,
        tuned=False,
        profile=profile,
        profile_dir=profile_dir,
        mod=mod,
        runs=runs,
    )


def evaluate_tuned_ansor(
    log_file,
    mod,
    params,
    device_info,
    test_inputs,
    target_outputs,
    use_ndk=False,
    dtype="float32",
    hardware_params=None,
    validate=True,
    limit_std=None,
    profile: Optional[bool] = False,
    profile_dir: Optional[os.PathLike] = "/tmp/tvmdbg",
    runs: Optional[int] = 10,
):
    timings = dict()
    runner, target, target_host = get_target(device_info)

    start = time.time()
    # Compile the whole network
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(
                mod, target=target, target_host=target_host, params=params
            )
    end = time.time()
    timings["compilation"] = end - start

    return _evaluate_model(
        lib,
        device_info,
        test_inputs,
        target_outputs,
        tuned=True,
        dtype=dtype,
        use_ndk=use_ndk,
        limit_std=limit_std,
        profile=profile,
        profile_dir=profile_dir,
        runs=runs,
        timings=timings,
    )


def evaluate_tuned_ansor_existing_cache(
    log_file,
    mod,
    params,
    device_info,
    test_inputs,
    target_outputs,
    use_ndk=False,
    dtype="float32",
    hardware_params=None,
    validate=True,
    limit_std=None,
    profile: Optional[bool] = False,
    profile_dir: Optional[os.PathLike] = "/tmp/tvmdbg",
    runs: Optional[int] = 10,
):
    timings = dict()
    runner, target, target_host = get_target(device_info)

    start = time.time()
    # Compile the whole network
    with auto_scheduler.ApplyHistoryBest(log_file):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
    end = time.time()
    timings["compilation"] = end - start

    return _evaluate_model(
        lib,
        device_info,
        test_inputs,
        target_outputs,
        tuned=True,
        dtype=dtype,
        use_ndk=use_ndk,
        limit_std=limit_std,
        profile=profile,
        profile_dir=profile_dir,
        runs=runs,
        timings=timings,
    )


def reject_outliers(data, m=1.5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def _evaluate_model(
    lib,
    device_info,
    test_inputs=None,
    target_output=None,
    dtype="float32",
    use_ndk=False,
    recheck=True,
    tuned=True,
    limit_std=None,
    profile: Optional[bool] = False,
    profile_dir: Optional[os.PathLike] = "/tmp/tvmdbg",
    mod: Optional[tvm.ir.module.IRModule] = None,
    runs: Optional[int] = 10,
    timings: Dict = dict(),
):
    # Create graph runtime
    if "remote" == device_info["host_type"]:
        if tuned:
            from tvm.auto_scheduler.utils import request_remote

            remote = request_remote(
                device_info["key"], device_info["address"], device_info["port"]
            )
        else:
            remote = rpc.connect(
                device_info["address"], device_info["port"], key=device_info["key"]
            )
        temp = utils.tempdir()

        if use_ndk:
            from tvm.contrib import ndk

            filename = "deploy_lib.so"
            path_lib = temp.relpath(filename)
            lib.export_library(path_lib, ndk.create_shared)
        else:
            filename = "deploy_lib.tar"
            path_lib = temp.relpath(filename)
            lib.export_library(path_lib)

        if "opencl" in device_info["target"]:
            dev = remote.cl()
        elif "cuda" in device_info["target"]:
            dev = remote.cuda(0)
        else:
            dev = remote.cpu(0)

        remote.upload(path_lib)
        loaded_lib = remote.load_module(filename)

    elif "local" == device_info["host_type"]:
        # dev = tvm.context(device_info["target_string"], 0)
        if "llvm" in device_info["target"]:
            dev = tvm.cpu(0)
        elif "cuda" in device_info["target"]:
            dev = tvm.cuda(0)
        else:
            raise ValueError(f"Unknwon host type: {device_info['host_type']}")
        loaded_lib = lib
    else:
        raise ValueError(f"Unknwon host type: {device_info['host_type']}")

    if profile:
        start = time.time()
        module = GraphModuleDebug(
            lib["debug_create"]("default", dev),
            [dev],
            lib.graph_json,
            dump_root=profile_dir,
        )
        end = time.time()
        timings["graph_module"] = end - start
    else:
        from tvm.contrib import graph_executor

        module = graph_executor.GraphModule(loaded_lib["default"](dev))

    print("Created graph runtime")
    # simport json

    # x = str(lib.graph_json)
    # with open("/tmp/g.json", "w") as f:
    #     json.dump(x, f, indent=2)

    print("Hey, number of test inputs is:", len(test_inputs))
    for i, (input_name, (test_input, in_dtype)) in enumerate(test_inputs.items()):
        test_input = np.array(test_input).astype(in_dtype)
        module.set_input(0, test_input)

    print("Set input data")
    start = time.time()
    if profile:
        results = []
        for _ in range(runs):
            module.run()
            res = os.path.join(profile_dir, "_tvmdbg_device_CPU_0", "trace.json")
            import json

            with open(res) as f:
                data = json.load(f)
            results.append(data)
        values = dict()
        for d in results:
            for n, t in d.items():
                if n in values:
                    values[n].append(t)
                else:
                    values[n] = [t]

        med = 0
        node_times = dict()
        for k, v in values.items():
            node_times[k] = np.median(v) / 1000
            med += node_times[k]
        std = None
        with open(
            os.path.join(profile_dir, "_tvmdbg_device_CPU_0", "formatted_trace.json"),
            "w",
        ) as f:
            json.dump(node_times, f, indent=2)
    else:
        # Evaluate
        # first test run
        print("Preparing first run")
        start = time.time()
        module.run()
        tvm_output = module.get_output(0).asnumpy()
        end = time.time()
        print("Did first run, took:", end - start)
        absolute_tolerance = 0.0001
        relative_tolerance = 0.0001
        if target_output is not None:
            # np.testing.assert_allclose(
            #     np.array(target_output).flatten(),
            #     tvm_output.flatten(),
            #     rtol=relative_tolerance,
            #     atol=absolute_tolerance,
            # )
            ...

        start = time.time()
        print(f"Evaluating, with {runs} runs")
        ftimer = module.module.time_evaluator("run", dev, number=3, repeat=runs)

        results = list(np.array(ftimer().results) * 1e3)  # convert to millisecond
        print("Collected data")
        end = time.time()
        print("Did full run, took:", end - start)
        out = module.get_output(0).asnumpy().flatten()
        med, std = np.median(results), np.std(results)

    end = time.time()
    timings["measure_and_collate"] = end - start
    print(timings)
    return med, std


def tvm_inference_subprocess(
    network_name: str,
    network_path: os.PathLike,
    device_name: str,
    logfile: os.PathLike,
    profiling: Optional[bool] = False,
    profiling_dir: Optional[os.PathLike] = None,
    runs: Optional[int] = 10,
):
    """Run the model and logfile as a subprocess.
    Necessary because sometimes the TVM runtime can crash
    in a way that cannot be easily recovered from via a try/except block

    :param network_name:
    :param network_path:
    :param device_name:
    :param logfile:
    :param profiling:
    :param profiling_dir:
    :param runs:
    :returns:

    """
    with tempfile.NamedTemporaryFile() as inf_data:
        my_args = [
            os.path.join(
                pathlib.Path(__file__).parent.absolute(),
                "../scripts/",
                "tvm_inference_script.py",
            ),
            "--model_path",
            network_path,
            "--model_name",
            network_name,
            "--device_name",
            device_name,
            "--log_file",
            logfile,
            "--output_file",
            inf_data.name,
            # "--runs",
            # runs,
        ]
        if profiling:
            my_args.append("--profile")
            my_args += ["--profile_dir", profiling_dir]
        p = subprocess.run(
            [
                "python3",
                *my_args,
            ]
        )
        if p.returncode == 0:
            with open(inf_data.name, "rb") as f:
                med_time, std_time = pickle.load(f)
            return med_time, std_time
        else:
            print(f"Error running")
            return None, None
