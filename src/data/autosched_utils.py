#!/usr/bin/env python
import numpy as np
import time
import multiprocessing
from typing import List
import shutil
import logging

from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.contrib import utils


def get_target(device_info):
    if "remote" == device_info["host_type"]:
        runner = auto_scheduler.RPCRunner(
            device_info["key"],
            device_info["rpc_address"],
            device_info["rpc_port"],
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


def tasks_tune(
    log_file,
    tasks,
    task_weights,
    device_info,
    ntrials,
    use_ndk=False,
    timeout=None,
    stop_points: List[float] = None,
    finetune: bool = False,
):
    if finetune is False:
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    else:
        tuner = auto_scheduler.TaskScheduler(
            tasks, task_weights, load_log_file=log_file
        )

    runner, target, target_host = get_target(device_info)

    if "sm_75" in target:
        from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

        set_cuda_target_arch("sm_75")

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=ntrials,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    start = time.time()
    p = multiprocessing.Process(
        target=tuner.tune, name="ansor_tune", args=(tune_option,)
    )
    p.start()

    if timeout is not None:
        # kill tuning after a given timeout
        print(f"We will stop early after {timeout} seconds")
        p.join(timeout)
        if p.is_alive():
            # Terminate foo
            p.terminate()
            p.join()
    elif stop_points is not None:
        # periodically make a copy of the tuning file
        print("running with stop points")
        i = 0
        time.sleep(stop_points[0])  # wait the first period
        try:
            shutil.copy(log_file, log_file + f".{i}.json")
        except:
            open(log_file + f".{i}.json", "a").close()

        logging.info(f"Making copy of file after {i} steps, {stop_points[i]} seconds")
        for i in range(1, len(stop_points)):
            wait = stop_points[i] - stop_points[i - 1]
            if wait > 0:
                time.sleep(wait)
            try:
                shutil.copy(log_file, log_file + f".{i}.json")
            except:
                open(log_file + f".{i}.json", "a").close()
            logging.info(
                f"Making copy of file after {i} steps, {stop_points[i]} seconds"
            )
        if p.is_alive():
            p.terminate()
            p.join()
    else:
        p.join()
    # tuner.tune(tune_option)
    tuning_time = time.time() - start

    return tuning_time


def tune_and_evaluate(
    log_file,
    mod,
    params,
    test_input_data,
    target_outputs,
    device_info,
    ntrials=200,
    use_ndk=False,
    hardware_params=None,
    dtype="float32",
    workload_pool=None,
    timeout: int = None,
    stop_points: List[float] = None,
):
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        device_info["target"],
        device_info["host"],
        hardware_params,
    )

    if len(tasks) < 1:
        raise ValueError("Expected to have at least one task to run")

    if workload_pool is not None:
        ntrials = workload_pool * len(tasks)
        print(f"We are doing a workload pool with {ntrials} ({len(tasks)} workloads)")

    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    print("Begin tuning...")

    args = [
        log_file,
        tasks,
        task_weights,
        device_info,
        ntrials,
        use_ndk,
        timeout,
        stop_points,
    ]
    tuning_time = tasks_tune(*args)

    print(f"Tuning time: {tuning_time}")

    med_time, std_time = compile_tuned_graph(
        log_file,
        mod,
        params,
        test_input_data,
        target_outputs,
        device_info,
        dtype="float32",
        use_ndk=False,
    )
    return med_time, std_time, tuning_time


def compile_tuned_graph(
    log_file,
    mod,
    params,
    test_inputs,
    target_outputs,
    device_info,
    dtype="float32",
    use_ndk=False,
):
    runner, target, target_host = get_target(device_info)

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(
                mod, target=target, target_host=target_host, params=params
            )

    # Create graph runtime
    if "remote" == device_info["host_type"]:
        print("=============== Request Remote ===============")
        from tvm.auto_scheduler.utils import request_remote

        remote = request_remote(
            device_info["key"], device_info["rpc_address"], device_info["rpc_port"]
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
            dev = remote.cuda()
        else:
            dev = remote.cpu(0)

        remote.upload(path_lib)
        loaded_lib = remote.load_module(filename)

    elif "local" == device_info["host_type"]:
        loaded_lib = lib
        if "opencl" in device_info["target"]:
            dev = tvm.cl()
        elif "cuda" in device_info["target"]:
            dev = tvm.cuda(0)
        else:
            dev = tvm.cpu(0)

    else:
        raise ValueError(f"Unknwon host type: {device_info['host_type']}")

    module = graph_executor.GraphModule(loaded_lib["default"](dev))

    for i, (input_name, (test_input, in_dtype)) in enumerate(test_inputs.items()):
        test_input = np.array(test_input).astype(in_dtype)
        module.set_input(0, test_input)

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", dev, repeat=3, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )
    return np.mean(prof_res), np.std(prof_res)
