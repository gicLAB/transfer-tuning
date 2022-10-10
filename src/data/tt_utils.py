#!/usr/bin/env python
import os
import ast
import numpy as np
import tempfile
import shutil
import logging
from tqdm import tqdm
from typing import List, Optional, Dict
import multiprocessing

from tvm import auto_scheduler
from tvm.auto_scheduler.search_task import SearchTask
from tvm.auto_scheduler import RecordReader
from tvm.auto_scheduler.measure import local_builder_build, MeasureInput, local_run
from tvm.relay.backend.compile_engine import get as get_compile_engine

from src.inference.tvm_inference import (
    tvm_inference_subprocess,
    evaluate_tuned_ansor_existing_cache,
)
from .workload_utils import get_tracing_names_2, get_cache_names
from src.scripts.utils import get_device


def estimate_iterations(
    wkl_ids,
    wkl_ids2,
    wkl_classes: dict,
    wkl_classes2: dict,
) -> int:
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


def measure_single_workload(inputs, repeats: int = 10, parallel: bool = False) -> float:
    logger = logging.getLogger("transfer-tuning")
    timeout, n_parallel = 3, multiprocessing.cpu_count() if parallel else 1
    build_results = local_builder_build(inputs, timeout, n_parallel)
    vs = []

    res = local_run(inputs, build_results, 10, repeats)

    vs = []
    for r in res:
        vs.append(r.costs[0].value)
        if r.error_no != 0:
            logger.info(f"Return code: {r.error_no}, {r.error_msg}, {r.costs[0].value}")

    cost = np.median(vs) * 1000  # return result in ms

    return cost, r.error_no, get_workload_dag_test(build_results)


def test_workload(
    logfile: os.PathLike, repeats: int = 1, parallel: bool = False, runs: int = 1
) -> float:
    """Measures the time for a single workload with a given autoschedule file in Ansor

    :param logfile: the path to the logfile containing workload and schedule information
    :returns: Inference time of workload

    """
    reader = RecordReader(logfile)
    inputs, _ = reader.read_lines()
    runs = 1
    print("Len inputs:", len(inputs), logfile)
    inputs = [inputs[0] for _ in range(runs)]

    return measure_single_workload(inputs, repeats, parallel)


def test_untuned_workload(task: SearchTask) -> float:
    """Measures the time for a single workload with the default schedule

    :param task: TVM auto-schedule search task

    :returns: Inference time of workload

    """
    inputs = MeasureInput(task, state=task.compute_dag.get_init_state())
    runs = 1
    inputs = [inputs for _ in range(runs)]

    return measure_single_workload(inputs)


def get_workload_dag_test(build_results):
    for r in build_results:
        accum = 1
        temp = 1
        if len(r.args) == 3:
            for i, a in enumerate(r.args):

                # print("shape:", a.shape)
                # print("prod:", np.prod(list(a.shape)))
                if i in [0, 2]:
                    accum *= a.shape[1]
                elif i == 1:
                    temp = np.prod(list(a.shape))
                else:
                    raise ValueError("We are the wrong size")
        break
    return accum != temp


def replace_wkl_id(
    orig_logfile: os.PathLike, new_wkl_id: str, new_logfile: os.PathLike, write_mode="w"
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


def generate_logfile_from_mapping(
    mapping: Dict[str, str],
    best_log_file: os.PathLike,
    log_file_dir: os.PathLike,
):
    open(best_log_file, "w").close()

    logfiles = []
    for wkl, tuning in mapping.items():
        if tuning is None:
            continue
        new_logf = os.path.join(log_file_dir, f"{wkl}_{tuning}.json")
        logfiles.append(new_logf)
    combine_log_files(logfiles, best_log_file)


def evaluate_mapping(
    mapping: Dict[str, str],
    log_file_dir: os.PathLike,
    network_name: str,
    network_path: os.PathLike,
    device_name: str,
    profiling: Optional[bool] = False,
    profiling_dir: Optional[os.PathLike] = None,
    runs: Optional[int] = 1,
):
    best_log_file = "/tmp/neo.log"

    timings = dict()
    import time

    start = time.time()
    generate_logfile_from_mapping(mapping, best_log_file, log_file_dir)
    end = time.time()
    timings["logfile_gen"] = end - start

    start = time.time()
    print("Hello world")
    print(
        network_name,
        network_path,
        device_name,
        best_log_file,
    )
    med_time, std_time = tvm_inference_subprocess(
        network_name,
        network_path,
        device_name,
        best_log_file,
        profiling=profiling,
        profiling_dir=profiling_dir,
        runs=runs,
    )
    end = time.time()
    timings["inf_time_collection"] = end - start
    print(timings)
    # if profiling:
    #     exit(1)
    return med_time, std_time


def evaluate_mapping_v2(
    mod,
    params,
    test_inputs,
    target_outputs,
    mapping: Dict[str, str],
    log_file_dir: os.PathLike,
    network_name: str,
    network_path: os.PathLike,
    device_name: str,
    profiling: Optional[bool] = False,
    profiling_dir: Optional[os.PathLike] = None,
    runs: Optional[int] = 1,
):
    best_log_file = "/tmp/neo.log"

    timings = dict()
    import time

    start = time.time()
    generate_logfile_from_mapping(mapping, best_log_file, log_file_dir)
    end = time.time()
    timings["logfile_gen"] = end - start

    start = time.time()

    device_info = get_device(device_name)

    try:
        med_time, std_time = evaluate_tuned_ansor_existing_cache(
            best_log_file,
            mod,
            params,
            device_info,
            test_inputs,
            target_outputs,
            profile=profiling,
            profile_dir=profiling_dir,
            runs=runs,
        )
    except Exception as e:
        print("Error running", e)
        med_time, std_time = None, None
    end = time.time()
    timings["inf_time_collection"] = end - start
    print(timings)

    return med_time, std_time


def get_wkl_profiling_inf_time(
    wkl_ids: List[str],
    wkl_full_params,
    tasks: List[auto_scheduler.SearchTask],
    task_weights: List[int],
    profile_dir: os.PathLike = "data/scratchpad/tvmdbg/_tvmdbg_device_CPU_0",
    drop_contrib_NCHWc=False,
):
    """Retrieve the inference times of workloads traced from a full model

    :param wkl_names:
    :returns:

    """
    profiling_names = get_tracing_names_2(
        profile_dir,
        wkl_full_params,
        tasks,
        task_weights,
        drop_contrib_NCHWc=drop_contrib_NCHWc,
    )
    res = os.path.join(profile_dir, "formatted_trace.json")
    import json

    with open(res) as f:
        inf_times = json.load(f)
    # res = os.path.join(profile_dir, "formatted_trace.csv")
    # df = pd.read_csv(res, sep="\t")
    # inf_times = dict()
    # for idx, row in df.iterrows():
    #     node_name = row["Node Name"]
    #     inf_time = row["Time(us)"]
    #     inf_times[node_name] = inf_time / 1000

    wkl_times = dict()
    for wkl, profile_names in profiling_names.items():
        # try except, because TVM has so many naming schemes
        # that have complex rules
        vs = []
        for p in profile_names:
            try:
                vs.append(inf_times[p])
            except Exception as e:
                print("Could not find {wkl}:{p}")
                raise e

        wkl_times[wkl] = np.median(vs)

    return wkl_times


def remove_wkl_from_tvm_build_cache(
    wkl_id, cache_names, tasks, task_weights, wkl_full_params
):
    target_func_names = cache_names[wkl_id]
    # gosh classes would really make this code much cleaner lol
    ceng = get_compile_engine()
    cache_items = ceng.items()

    count = 0
    func_names = []
    for k, v in cache_items:
        if v.cached_func is None:
            continue
        func_name = v.cached_func.func_name
        func_names.append(func_name)
        if func_name in target_func_names:
            ceng.cache_remove(k)
            count += 1
    if count != len(target_func_names):
        print(
            f"We have a problem, we should have removed",
            f"{len(target_func_names)} from the cache",
            f"but instead we only removed {count}",
        )
        print("func_names:", func_names)
        print("target_func_names:", target_func_names)
        # it's possible the cache names have changes
        cache_names = get_cache_names(tasks, task_weights, wkl_full_params)
        if cache_names[wkl_id] in func_names:
            print("Great, so it's just the cache changed")
            remove_wkl_from_tvm_build_cache(
                wkl_id, cache_names, tasks, task_weights, wkl_full_params
            )
    else:
        print("Cache remove success!!")
    return cache_names


def full_selection_policy_1(
    mod,
    params,
    test_inputs,
    target_outputs,
    current_best: float,
    tuned_wkl_times: Dict[str, float],
    standalone_predictions: Dict[str, Dict[str, float]],
    best_mapping: Dict[str, str],
    split_schedule_dir: os.PathLike,
    network_name: str,
    network_path: os.PathLike,
    device_name: str,
    full_evaluations: int,
    wkl_names: Dict[str, str],
    profiling_names,
    cache_names,
    tasks,
    task_weights,
    wkl_full_params,
):
    orig_best = current_best.copy()
    logger = logging.getLogger("transfer-tuning")
    t = tqdm(total=full_evaluations)

    # go through all the tuned workloads
    # see if the performance in standalone matches the actual performance
    # if not, we should check if we can do better
    deltas = dict()  # for each workload, the difference between the standalone time and
    # the time in the full model
    total_full_time = 0
    total_standalone_time = 0

    print("standalone_predictions:", standalone_predictions)

    for wkl_id, inf_time in tuned_wkl_times.items():
        wkl_name = wkl_names[wkl_id]
        total_full_time += inf_time
        tuned_wkl_id = best_mapping[wkl_id]
        if tuned_wkl_id is None:
            tuned_wkl_id = "Untuned"
        standalone_time = standalone_predictions[wkl_name][tuned_wkl_id]
        total_standalone_time += standalone_time

        if inf_time > standalone_time:
            logger.info(f"{inf_time} > {standalone_time}")
            deltas[wkl_id] = inf_time - standalone_time
    logger.info(f"Total full time: {total_full_time}")
    logger.info(f"Total standalone time: {total_full_time}")

    # sort deltas by biggest to smallest, and decide how many full evals each workload gets
    deltas = {k: v for k, v in sorted(deltas.items(), key=lambda item: item[1])}
    total_deltas = sum(deltas.values())
    deltas_prop = {
        k: round(full_evaluations * (v / total_deltas)) for k, v in deltas.items()
    }
    logger.info(f"Deltas: {deltas}")
    logger.info(f"Delta props: {deltas_prop}")

    # choose the workload/schedule pairs we
    carry = 0  # extra tries from previous workload
    candidates = {k: [] for k, _ in tuned_wkl_times.items()}
    total_candidates = 0
    for wkl_id, tries in deltas_prop.items():
        wkl_name = wkl_names[wkl_id]
        tries += carry
        preds = standalone_predictions[wkl_name]
        tuned_wkl = best_mapping[wkl_id]
        if tuned_wkl is None:
            tuned_wkl = "Untuned"
        del preds[tuned_wkl]  # remove the one we already know
        # remove ones with unreasonably high times
        rms = []
        for tuned_wkl, wkl_time in preds.items():
            if wkl_time > 10e9:
                if tuned_wkl is None:
                    tuned_wkl = "Untuned"
                rms.append(tuned_wkl)
        for r in rms:
            del preds[r]

        if tries > len(preds):
            carry = len(preds) - tries
            tries = len(preds)
        else:
            carry = 0

        candidates[wkl_id] = list(standalone_predictions[wkl_name].keys())[:tries]
        total_candidates += tries
        assert len(candidates[wkl_id]) == tries

    # if we have some candidates left-over (i.e. standalone prediction was okay)
    tmp = total_candidates
    if total_candidates < full_evaluations:
        remaining_evals = full_evaluations - total_candidates
        remaining_evals_count = remaining_evals
        # allocate get workloads full evaluations based on their contribution to the
        # overall inference time
        wkl_prop = {
            k: int(np.ceil(remaining_evals * (inf_time / orig_best)))
            for k, inf_time in tuned_wkl_times.items()
        }
        print("Oh no, we have fewer: ", total_candidates, full_evaluations)
        carry = 0
        for wkl_id, tries in wkl_prop.items():
            if remaining_evals_count <= 0:
                break
            wkl_name = wkl_names[wkl_id]
            tries += carry
            num_so_far = len(candidates[wkl_id])
            tuned_wkl = best_mapping[wkl_id]
            if tuned_wkl is None:
                tuned_wkl = "Untuned"
            preds = standalone_predictions[wkl_name]
            try:
                del preds[tuned_wkl]  # remove the one we already know
            except:
                ...
            if num_so_far >= len(preds):
                # we are already going to check every schedule for this workload
                carry = tries
                continue
            extra_checks = list(standalone_predictions[wkl_name].keys())[num_so_far:]
            if len(extra_checks) < tries:
                carry = tries - len(extra_checks)
                tries = len(extra_checks)
            try:
                extra_candidates = extra_checks[:tries]
            except Exception as e:
                print("hey, tries is", tries, type(tries))
                print("extra checks is:", extra_checks)
            candidates[wkl_id] += extra_candidates
            total_candidates += len(extra_candidates)
            remaining_evals_count -= len(extra_candidates)

    # print("Hey, our candidates are:", candidates[wkl_id], total_candidates)
    # print(wkl_prop)
    # print(full_evaluations, tmp)

    # for k, inf_time in tuned_wkl_times.items():
    #     print(
    #         k,
    #         inf_time,
    #         orig_best,
    #         (inf_time / orig_best),
    #         remaining_evals * (inf_time / orig_best),
    #     )

    # tot = 0
    # for wkl_id, cands in candidates.items():
    #     for tuning in cands:
    #         tot += 1
    # print("Hey our tot will be:", tot)
    # print(candidates)
    # exit(1)
    actual_runs = 0
    for wkl_id, cands in candidates.items():
        wkl_name = wkl_names[wkl_id]
        mapping = best_mapping.copy()
        for tuning in cands:
            if tuning == "Untuned":
                tuning = None
            mapping[wkl_id] = tuning

            # remove the target workload from the build cache
            cache_names = remove_wkl_from_tvm_build_cache(
                wkl_id, cache_names, tasks, task_weights, wkl_full_params
            )

            std_time, _ = evaluate_mapping_v2(
                mod,
                params,
                test_inputs,
                target_outputs,
                mapping,
                split_schedule_dir,
                network_name,
                network_path,
                device_name,
                profiling=False,  # not doing profiling here, perhaps in another policy
                runs=10,
            )
            actual_runs += 1
            t.update(1)
            if std_time is None:
                # remove the offending compiled workload from the cache
                remove_wkl_from_tvm_build_cache(
                    wkl_id, cache_names, tasks, task_weights, wkl_full_params
                )
                continue
            elif std_time < current_best:
                logging.info(f"New best time found: {std_time}")
                best_mapping = mapping.copy()
                current_best = std_time

    logging.info(f"Best time is: {current_best}, previous was: {orig_best}")
    print(f"Acutal runs: {actual_runs} + 1")
    if actual_runs < full_evaluations:
        print("Aw naw, we have got a problem", actual_runs, full_evaluations)
    return current_best, best_mapping
