#!/usr/bin/env python
import os
import json
import ast
import shutil
import tempfile
from operator import itemgetter
import time
from typing import List, Tuple, Optional, Union, Dict
from hashlib import sha256
import numpy as np
import pandas as pd
import string
from tqdm import tqdm
from collections import Counter
from tvm import auto_scheduler
from tvm.auto_scheduler import RecordReader
from tvm.auto_scheduler.measure import local_builder_build, MeasureInput
from tvm.auto_scheduler.search_task import SearchTask
import multiprocessing
from .workload_utils import get_dag_fn
from ..inference.tvm_inference import tvm_inference_subprocess


def logfile_splitter(logfiles: List[os.PathLike], split_dir: os.PathLike):
    """Split an Ansor logfile in separate logfiles for each workload id

    :param logfile:
    :param split_dir:
    :returns:

    """

    wkl_ids = set()
    files = set()
    for logfile in logfiles:
        with open(logfile) as f:
            content = f.readlines()
            for i, line in enumerate(content):
                data = json.loads(line)
                long_wkl_id = data["i"][0][0]
                id_sha = sha256(long_wkl_id.encode("utf-8")).hexdigest()
                wkl_ids.add(id_sha)
                fname = os.path.join(split_dir, f"{id_sha}.json")
                files.add(fname)
                with open(fname, "a") as f:
                    f.write(line)
    return list(wkl_ids), list(files)


def replace_wkl_id(
    orig_logfile: os.PathLike,
    new_wkl_id: str,
    new_logfile: os.PathLike,
    write_mode="w",
    single_line=False,
):
    """Replace the workload id in an Ansor logfile and write to new file

    :param orig_logfile:
    :param new_wkl_id:
    :param new_logfile:
    :param write_mode:
    :returns:

    """

    with open(orig_logfile, "r") as f:
        data = f.readlines()
        new_lines = []
        for d in data:
            workload = ast.literal_eval(d)
            workload["i"][0][0] = new_wkl_id
            new_lines.append(workload)
            if single_line:
                break

    # add workload to new file completely
    with open(new_logfile, write_mode) as f:
        for l in new_lines:
            wkl = str(repr(l))
            wkl = wkl.replace('"', '\\"')
            wkl = wkl.replace("'", '"')
            f.write(wkl + "\n")


def transfer_tune_log_file(
    new_wkl_id: str,
    old_logfile: os.PathLike,
    new_logfile: os.PathLike,
):
    replace_wkl_id(old_logfile, new_wkl_id, new_logfile, write_mode="w")


def combine_log_files(
    log_files: List[os.PathLike],
    new_logfile: os.PathLike,
):
    """Combine a list of logfiles into a new file

    :param log_files:
    :param new_logfile:
    :returns:

    """

    with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
        if new_logfile in log_files:
            # make a copy if the target is also a source
            log_files.remove(new_logfile)
            shutil.copy(new_logfile, temp_file.name)
            log_files.append(temp_file.name)
        with open(new_logfile, "w") as outfile:
            for fname in log_files:
                if fname is None:
                    print("Error:", log_files)
                with open(fname) as infile:
                    outfile.write(infile.read())


def estimate_iterations(
    wkl_ids,
    wkl_ids2,
    wkl_classes: dict,
    wkl_classes2: dict,
):
    """Estimate the number of valid workload/tuning pairs that will need to be evaluated

    :param wkl_ids:
    :param wkl_ids2:
    :param wkl_classes:
    :param wkl_classes2:
    :returns:

    """

    total = 0
    for wkl_id in wkl_ids:
        total += 1  # for running standalone
        for wkl_id2 in wkl_ids2:
            class1 = wkl_classes[wkl_id]
            class2 = wkl_classes2[wkl_id2]
            if class1 != class2:
                continue
            else:
                total += 1
    return total


def measure_single_workload(inputs, repeats: int = 1, parallel: bool = True) -> float:
    timeout, n_parallel = 3, multiprocessing.cpu_count() if parallel else 1
    results = local_builder_build(inputs, timeout, n_parallel, repeat=repeats)
    vs = []
    for r in results:
        if (r.error_no == 0) or (r.error_no == 6):
            # acceptable exit code
            vs.append(r.time_cost)
        elif r.error_no != 2:
            # unexpected exit code
            print("Unexpected error code:", r.error_no)
            return timeout * 2
            # exit(1)
        else:
            # if we have an error pretend the cost is too high
            return timeout * 2

    cost = np.median(vs)
    return cost


def test_workload(
    logfile: os.PathLike, repeats: int = 1, parallel: bool = True, runs: int = 5
) -> float:
    """Measures the time for a single workload with a given autoschedule file in Ansor

    :param logfile: the path to the logfile containing workload and schedule information
    :returns: Inference time of workload

    """
    reader = RecordReader(logfile)
    inputs, _ = reader.read_lines()
    inputs = [inputs[0] for _ in range(runs)]

    return measure_single_workload(inputs, repeats, parallel)


def test_untuned_workload(task: SearchTask) -> float:
    """Measures the time for a single workload with the default schedule

    :param task: TVM auto-schedule search task

    :returns: Inference time of workload

    """
    inputs = MeasureInput(task, state=task.compute_dag.get_init_state())
    runs = 5
    inputs = [inputs for _ in range(runs)]

    return measure_single_workload(inputs)


def get_conv2d_macs(params):
    in_w = params["in_shape"][2]
    try:
        stride = params["strides"][0]
    except:
        print('Param not found "params":', params)
        return
    kdim = params["kernel_shape"][-1]
    padding = params["padding"][-1]
    out_w = int((in_w + 2 * padding - kdim) / stride + 1)
    out_c = params["kernel_shape"][0]
    macs = np.prod(params["kernel_shape"]) * out_w * out_w
    if "bias" in params["class"]:
        macs += out_c
    if "relu" in params["class"]:
        macs += out_c
    return macs


def get_workload_info_v2(
    mod,
    params,
    device_info,
    verbose=False,
    skip_error=False,
    hardware_params=None,
):
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        device_info["target"],
        device_info["host"],
        hardware_params,
    )

    workload_classes = dict()
    full_wkl_ids = dict()  # deal with TVMs new verbose format
    all_params = dict()
    for i, tsk in enumerate(tasks):
        tgt_wkl_id = tsk.workload_key
        tgt_dag = str(tsk.compute_dag)
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()
        tgt_desc = tsk.desc

        fn = get_dag_fn(tgt_dag, tgt_desc)
        if fn is None:
            if verbose:
                print("Unknown dag")
                print(tgt_dag)
            if not skip_error:
                raise ValueError("Unknown Dag", tgt_dag)
            continue
        if verbose:
            print(f"\n\nChecking workload{tgt_wkl_id}"), print(tgt_dag)
        net_in_shape_dict, workload_class, params = fn(tgt_dag)

        wkl_id = tgt_wkl_id[2:34]
        params["wkl_id"] = wkl_id
        params["wkl_class"] = workload_class
        if "onv2d" in tgt_dag:
            params["size"] = np.prod(params["in_shape"]) * np.prod(
                params["kernel_shape"]
            )
            if "conv2d_NCHWc" in tgt_dag:
                params["macs"] = get_conv2d_macs(params)
        elif "dense" in tgt_dag:
            params["size"] = params["units"] * np.prod(params["in_shape"])
        else:
            params["size"] = np.prod(params["in_shape"])
        all_params[tgt_wkl_id_sha] = params
        workload_classes[tgt_wkl_id_sha] = workload_class
        full_wkl_ids[tgt_wkl_id_sha] = tgt_wkl_id
    return workload_classes, full_wkl_ids, all_params


def get_workload_info_v3(
    tasks,
    verbose=False,
    skip_error=False,
):
    workload_classes = dict()
    full_wkl_ids = dict()  # deal with TVMs new verbose format
    all_params = dict()
    for i, tsk in enumerate(tasks):
        tgt_wkl_id = tsk.workload_key
        tgt_dag = str(tsk.compute_dag)
        tgt_wkl_id_sha = sha256(tgt_wkl_id.encode("utf-8")).hexdigest()
        tsk_desc = tsk.desc
        fn = get_dag_fn(tgt_dag)
        if fn is None:
            if verbose:
                print("Unknown dag")
                print(tgt_dag)
            if not skip_error:
                raise ValueError("Unknown Dag", tgt_dag)
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
            if "conv2d_NCHWc" in tgt_dag:
                params["macs"] = get_conv2d_macs(params)
        elif "dense" in tgt_dag:
            params["size"] = params["units"] * np.prod(params["in_shape"])
        else:
            params["size"] = np.prod(params["in_shape"])
        all_params[tgt_wkl_id_sha] = params
        workload_classes[tgt_wkl_id_sha] = workload_class
        full_wkl_ids[tgt_wkl_id_sha] = tgt_wkl_id
    return workload_classes, full_wkl_ids, all_params


def get_workload_info(
    tasks: List[auto_scheduler.SearchTask],
    verbose: bool = False,
    skip_error: bool = False,
):
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


def get_top(
    replacements: Tuple[str, float],
    threshold: float = 1.3,
    truncate: Optional[int] = None,
) -> Union[str, List[str]]:
    """Return best replacement and top alternatives"""
    if len(replacements) < 1:
        return None, []

    top_tuning = replacements[0][0]
    top_time = replacements[0][1]
    candidates = []

    for i, (r, t, _) in enumerate(replacements):
        # if truncate is not None and i > truncate:
        #     # limit the number of alternatives we consider for each workload
        #     break
        # if t <= top_time * threshold:
        candidates.append(r)
        # else:
        #     break
    if len(candidates) == 1:
        # we won't evaluate the top candidate twice
        return top_tuning, []
    else:
        return top_tuning, candidates[1:]


def evaluate_replacements(
    mapping: Dict[str, str],
    log_file_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
    profiling: Optional[bool] = False,
):
    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    logfiles = []
    for wkl, tuning in mapping.items():
        if tuning is None:
            continue
        new_logf = os.path.join(log_file_dir, f"{wkl}_{tuning}.json")
        logfiles.append(new_logf)
    combine_log_files(logfiles, best_log_file)
    med_time, std_time = tvm_inference_subprocess(
        network_file, device_name, best_log_file, profiling=profiling
    )
    return med_time, std_time


def evaluate_mapping(
    mapping: Dict[str, str],
    log_file_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
):
    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    logfiles = []
    for wkl, tuning in mapping.items():
        if tuning is None:
            continue
        new_logf = os.path.join(log_file_dir, f"{wkl}_{tuning}.json")
        logfiles.append(new_logf)
    combine_log_files(logfiles, best_log_file)
    med_time, std_time = tvm_inference_subprocess(
        network_file, device_name, best_log_file
    )
    return med_time, std_time


def evaluate_alternatives(
    current_replacement_map: Dict[str, str],
    candidate_pairs,
    baseline_inf_time: float,
    logfile_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
):
    pairs = 0
    for wkl, tunings in candidate_pairs.items():
        pairs += len(tunings)
    t = tqdm(total=pairs)
    current_best = baseline_inf_time
    for wkl, tunings in candidate_pairs.items():
        for tuning in tunings:
            test_map = current_replacement_map.copy()
            test_map[wkl] = tuning
            std_time, _ = evaluate_replacements(
                test_map,
                logfile_dir,
                network_file,
                device_name,
            )
            t.update(1)
            if std_time is None:
                continue
            elif std_time < current_best:
                print("New best time found:", std_time)
                current_replacement_map = test_map.copy()
                current_best = std_time
    print("Previous best_time:", baseline_inf_time, "New best time:", current_best)
    return current_best, current_replacement_map


def evaluate_weighted_random_alternatives(
    current_replacement_map: Dict[str, str],
    candidate_times,
    baseline_inf_time: float,
    logfile_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
    task_weights: List[int],
    num_allowed: int = 20,
):
    pairs = 0
    p_dists = dict()  # weightings for schedules to try for each workload
    average_times = []
    for i, (wkl, tunings) in enumerate(candidate_times.items()):
        wkl_weight = task_weights[i]
        times = []
        inverse_times = []  # we want lower times to be rated higher
        for (tuning_id, wkl_time, _) in tunings:
            inverse_times.append(3.0 - wkl_time)
            times.append(wkl_time)
        average_time = np.median(times)
        average_times.append(average_time * wkl_weight)
        p_dist = np.exp(inverse_times) / sum(
            np.exp(inverse_times)
        )  # get softmax of inverse of inference time
        p_dists[wkl] = p_dist

    wkl_p_dist = np.exp(average_times) / sum(
        np.exp(average_times)
    )  # get probability of sampling a workload

    choices = np.random.choice(
        list(candidate_times.keys()), num_allowed, replace=True, p=wkl_p_dist
    )
    wkl_sample_freq = Counter(choices)

    t = tqdm(total=num_allowed)

    current_best = baseline_inf_time
    for wkl, freq in wkl_sample_freq.items():
        options = [x[0] for x in candidate_times[wkl]]
        freq = min(freq, len(options))  # we can't measure schedules than we have
        choices = np.random.choice(options, freq, replace=False, p=p_dists[wkl])
        for c in choices:
            test_map = current_replacement_map.copy()
            test_map[wkl] = c
            std_time, _ = evaluate_replacements(
                test_map,
                logfile_dir,
                network_file,
                device_name,
            )
            t.update(1)
            if std_time is None:
                continue
            elif std_time < current_best:
                print("New best time found:", std_time)
                current_replacement_map = test_map.copy()
                current_best = std_time
    print("Previous best_time:", baseline_inf_time, "New best time:", current_best)
    return current_best, current_replacement_map


def evaluate_weighted_fixed_alternatives(
    current_replacement_map: Dict[str, str],
    candidate_times,
    baseline_inf_time: float,
    logfile_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
    task_weights: List[int],
    num_allowed: int = 20,
    return_more_data: bool = False,
):
    non_bad_pairs = 0
    average_times = []
    pruned_tunings = dict()
    print("Running fixed")

    for i, (wkl, tunings) in enumerate(candidate_times.items()):
        wkl_weight = task_weights[i]
        times = []
        pruned_tunings[wkl] = []
        for (tuning_id, wkl_time, _) in tunings:  # skip the best
            if wkl_time > 3.0:
                continue
            pruned_tunings[wkl].append(tuning_id)
            times.append(wkl_time)
            non_bad_pairs += 1
        if len(times) == 0:
            times = [0]
        average_time = np.mean(times)
        average_times.append(average_time * wkl_weight)

    print("Got here")
    wkl_p_dist = np.exp(average_times) / sum(
        np.exp(average_times)
    )  # get probability of sampling a workload
    #
    # num_allowed = min(num_allowed, non_bad_pairs)
    print("We have ", num_allowed, non_bad_pairs)
    num_tries = [round(num_allowed * prob) for prob in wkl_p_dist]
    meta_lst = list(enumerate(num_tries))

    sorted_num_tries = sorted(meta_lst, key=itemgetter(1), reverse=True)
    wkls = list(candidate_times.keys())
    t = tqdm(total=num_allowed)

    current_best = baseline_inf_time
    carry_forward = 0  # if we did not have enough schedules, carry forward
    print("before we start running, how do we eval")
    print("sorted_num_tries", sorted_num_tries)
    print("wkls", wkls)
    print(pruned_tunings)

    for i, num_try in sorted_num_tries:
        wkl = wkls[i]
        num_try += carry_forward
        if num_try > len(pruned_tunings[wkl]):
            carry_forward = num_try - len(pruned_tunings[wkl])
            num_try = len(pruned_tunings[wkl])
        else:
            carry_forward = 0
        for s in range(num_try):
            test_map = current_replacement_map.copy()
            test_map[wkl] = pruned_tunings[wkl][0]
            pruned_tunings[wkl].pop(0)  # remove the first item
            std_time, _ = evaluate_replacements(
                test_map,
                logfile_dir,
                network_file,
                device_name,
            )
            t.update(1)
            if std_time is None:
                continue
            elif std_time < current_best:
                print("New best time found:", std_time)
                current_replacement_map = test_map.copy()
                current_best = std_time
    print("Previous best_time:", baseline_inf_time, "New best time:", current_best)
    if not return_more_data:
        return current_best, current_replacement_map
    else:
        return current_best, current_replacement_map, non_bad_pairs


def evaluate_weighted_fixed_alternatives_neo(
    current_replacement_map: Dict[str, str],
    candidate_times,
    untuned_inf_time: float,
    baseline_inf_time: float,
    logfile_dir: os.PathLike,
    network_file: os.PathLike,
    device_name: str,
    task_weights: List[int],
    num_allowed: int = 20,
    return_more_data: bool = False,
):
    non_bad_pairs = 0
    average_times = []
    pruned_tunings = dict()
    print("Running fixed neo")
    current_best = baseline_inf_time
    bad_best = baseline_inf_time
    if 1.1 * untuned_inf_time < baseline_inf_time:
        current_best = untuned_inf_time * 1.1
        print("we we need to kill a bad time")
        # we need to kill a bad time
        for i, (wkl, tunings) in enumerate(candidate_times.items()):
            print(f"Hey {wkl}")
            if current_replacement_map[wkl] is None:
                continue
            else:
                print("Trying with nothing")
                num_allowed -= 1
                test_map = current_replacement_map.copy()
                current = test_map[wkl]
                test_map[wkl] = None
                std_time, _ = evaluate_replacements(
                    test_map,
                    logfile_dir,
                    network_file,
                    device_name,
                )
                if std_time is None:
                    continue
                elif std_time < bad_best:
                    print("New best time found:", std_time)
                    print("the bad schedule was:", current)
                    current_replacement_map = test_map.copy()
                    bad_best = std_time
                    if std_time < current_best:
                        print("we are free of this curse")
                        current_best = std_time
                        # exit(1)
                        break

    print("Did we make it?")
    for i, (wkl, tunings) in enumerate(candidate_times.items()):
        wkl_weight = task_weights[i]
        times = []
        pruned_tunings[wkl] = []
        for (tuning_id, wkl_time, _) in tunings:  # skip the best
            if wkl_time > 3.0:
                continue
            pruned_tunings[wkl].append(tuning_id)
            times.append(wkl_time)
            non_bad_pairs += 1
        if len(times) == 0:
            times = [0]
        average_time = np.mean(times)
        average_times.append(average_time * wkl_weight)

    print("Got here")
    wkl_p_dist = np.exp(average_times) / sum(
        np.exp(average_times)
    )  # get probability of sampling a workload
    #
    # num_allowed = min(num_allowed, non_bad_pairs)
    print("We have ", num_allowed, non_bad_pairs)
    num_tries = [round(num_allowed * prob) for prob in wkl_p_dist]
    meta_lst = list(enumerate(num_tries))

    sorted_num_tries = sorted(meta_lst, key=itemgetter(1), reverse=True)
    wkls = list(candidate_times.keys())
    t = tqdm(total=num_allowed)

    carry_forward = 0  # if we did not have enough schedules, carry forward
    print("before we start running, how do we eval")
    print("sorted_num_tries", sorted_num_tries)
    print("wkls", wkls)
    print(pruned_tunings)

    for i, num_try in sorted_num_tries:
        wkl = wkls[i]
        num_try += carry_forward
        if num_try > len(pruned_tunings[wkl]):
            carry_forward = num_try - len(pruned_tunings[wkl])
            num_try = len(pruned_tunings[wkl])
        else:
            carry_forward = 0
        for s in range(num_try):
            test_map = current_replacement_map.copy()
            test_map[wkl] = pruned_tunings[wkl][0]
            pruned_tunings[wkl].pop(0)  # remove the first item
            std_time, _ = evaluate_replacements(
                test_map,
                logfile_dir,
                network_file,
                device_name,
            )
            t.update(1)
            if std_time is None:
                continue
            elif std_time < current_best:
                print("New best time found:", std_time)
                current_replacement_map = test_map.copy()
                current_best = std_time
    print("Previous best_time:", baseline_inf_time, "New best time:", current_best)
    return current_best, current_replacement_map, non_bad_pairs


def across_network_oracle_presplit(
    base_time: float,
    device_name: str,
    tasks: List[SearchTask],
    wkl_classes: dict,
    wkl_classes_tuned: dict,
    full_wkl_ids: dict,
    network_name: str,
    network_file: os.PathLike,
    tuned_logfile_dir: os.PathLike,
    threshold: Optional[float] = 1.3,
    task_weights: Optional[List[int]] = None,
):
    model_wkl_ids = list(wkl_classes.keys())
    tuned_wkl_ids = list(wkl_classes_tuned.keys())

    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    replacements = dict()
    # mapping = dict()
    wkl_times = dict()
    candidates = dict()
    top_mapping = dict()

    vals = (string.ascii_uppercase + string.ascii_lowercase)[: len(tuned_wkl_ids)]
    tuned_id_alias = {w: n for w, n in zip(tuned_wkl_ids, vals)}

    tuned_temp_dir = tuned_logfile_dir

    num_iterations = estimate_iterations(
        model_wkl_ids,
        tuned_wkl_ids,
        wkl_classes,
        wkl_classes_tuned,
    )
    t = tqdm(total=num_iterations)
    x = 0
    # for each workload in the new network, find the best tuned workload in the old logfile to use
    for i, wkl_id in enumerate(model_wkl_ids):
        replacements[wkl_id] = []
        wkl_times[i] = []

        # get time for workload with no tuning
        wkl_time = test_untuned_workload(tasks[i])

        wkl_times[i].append(("No Tuning", wkl_time))
        replacements[wkl_id].append((None, wkl_time, wkl_id))
        t.update(1)

        for tuned_wkl_id in tuned_wkl_ids:
            if wkl_classes[wkl_id] != wkl_classes_tuned[tuned_wkl_id]:
                # skip if we are a different workload class
                continue
            # if wkl_id != tuned_wkl_id:
            #     # testing purposes focus on native tuning
            #     continue
            tuned_log_file = os.path.join(tuned_temp_dir, f"{tuned_wkl_id}.json")
            new_logf = os.path.join(tuned_temp_dir, f"{wkl_id}_{tuned_wkl_id}.json")

            if not os.path.exists(tuned_log_file):
                t.update(1)
                x += 1
                continue
            transfer_tune_log_file(
                full_wkl_ids[wkl_id],
                tuned_log_file,
                new_logf,
            )
            try:
                wkl_label = tuned_id_alias[tuned_wkl_id]
            except:
                wkl_label = tuned_wkl_id
                tuned_id_alias[tuned_wkl_id] = tuned_wkl_id

            wkl_time = test_workload(new_logf)

            wkl_times[i].append((wkl_label, wkl_time))
            x += 1
            t.update(1)
            if wkl_time == None:
                continue
            else:
                replacements[wkl_id].append((tuned_wkl_id, wkl_time, wkl_id))

        ##########
        # get candidate standalone workload/tuning pairs to evaluate in the full model
        replacements[wkl_id].sort(key=lambda x: x[1])
        top_mapping[wkl_id], candidates[wkl_id] = get_top(
            replacements[wkl_id], threshold=threshold
        )

    t.close()  # close the progress bar

    # create and evaluate the best logfile
    first_past_best_time, std_time = evaluate_replacements(
        top_mapping,
        tuned_temp_dir,
        network_file,
        device_name,
    )
    print(f"Current best time is {first_past_best_time}")
    # exit(1)
    first_pass_tt_end_time = time.time()
    # try alternatives:
    #

    if threshold > 0:
        (
            best_time,
            final_mapping,
            non_bad_pairs,
        ) = evaluate_weighted_fixed_alternatives_neo(
            top_mapping,
            replacements,
            base_time,
            first_past_best_time,
            tuned_temp_dir,
            network_file,
            device_name,
            task_weights,
            int(threshold),
            return_more_data=True,
        )
    else:
        best_time, final_mapping, non_bad_pairs = first_past_best_time, top_mapping, 0

    print_mapping = dict()
    tuned_id_alias[None] = None
    for i, wkl_id in enumerate(model_wkl_ids):
        if wkl_id in top_mapping:
            print_mapping[i] = tuned_id_alias[top_mapping[wkl_id]]
            print_mapping[i] = tuned_id_alias[final_mapping[wkl_id]]
        else:
            print_mapping[i] = None
    print(print_mapping)
    print(wkl_times)
    print(f"We had {non_bad_pairs}, best time: {best_time}")
    return (
        best_time,
        print_mapping,
        num_iterations,
        replacements,
        first_past_best_time,
        first_pass_tt_end_time,
        final_mapping,
    )


def classes_to_node_name(classes: List[str]):
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


def get_wkl_profiling_inf_time(wkl_names: List[str]):
    """Retrieve the inference times of workloads traced from a full model

    :param wkl_names:
    :returns:

    """

    profile_dir = "data/scratchpad/tvmdbg/_tvmdbg_device_CPU_0"
    res = os.path.join(profile_dir, "formatted_trace.csv")
    df = pd.read_csv(res, sep="\t")
    inf_times = dict()
    for idx, row in df.iterrows():
        node_name = row["Node Name"]
        inf_time = row["Time(us)"]
        inf_times[node_name] = inf_time / 1000

    wkl_times = dict()
    for wkl in wkl_names:
        wkl_times[wkl] = inf_times[wkl]
    return wkl_times
