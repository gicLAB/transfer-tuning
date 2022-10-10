import os
import logging
import shutil
import tempfile
from tqdm import tqdm
from typing import List, Optional, Dict
import tvm
import time
from tvm.auto_scheduler.search_task import SearchTask
from tvm.relay.backend.compile_engine import get as get_compile_engine
from .tt_utils import (
    estimate_iterations,
    test_untuned_workload,
    transfer_tune_log_file,
    test_workload,
    evaluate_mapping,
    evaluate_mapping_v2,
    get_wkl_profiling_inf_time,
    full_selection_policy_1,
)
from src.data.workload_utils import get_tracing_names_2, get_cache_names


def tt_internetwork_main(
    mod,
    params,
    test_inputs,
    target_outputs,
    device_name: str,
    tasks: List[SearchTask],
    wkl_classes: dict,
    wkl_classes_tuned: dict,
    wkl_names: Dict[str, str],
    full_wkl_ids: dict,
    network_name: str,
    network_path: os.PathLike,
    tuned_logfile_dir: os.PathLike,
    full_evaluations: Optional[int] = 10,
    task_weights: Optional[List[int]] = None,
    wkl_full_params=None,
    save_standalone_info=False,
    disable_conv2d=False,
):
    # print("Hey wkl_names", wkl_names)
    # exit(1)
    timings = {}
    model_wkl_ids = list(wkl_classes.keys())
    tuned_wkl_ids = list(wkl_classes_tuned.keys())

    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    tuned_temp_dir = tempfile.mkdtemp()
    mapping = dict()
    num_iterations = estimate_iterations(
        model_wkl_ids,
        tuned_wkl_ids,
        wkl_classes,
        wkl_classes_tuned,
    )
    wkl_standalone_times = dict()
    t = tqdm(total=num_iterations)
    x = 0

    import time

    # print(wkl_names)
    # exit(1)
    start = time.time()
    # for each workload in the new network, find the best tuned workload in the old logfile to use
    for i, wkl_id in enumerate(model_wkl_ids):
        wkl_name = wkl_names[wkl_id]

        wkl_standalone_times[wkl_name] = {}

        # get time for workload with no tuning
        # wkl_standalone_times[wkl_name]["Untuned"] = untuned_wkl_inf_times[wkl_name]
        wkl_standalone_times[wkl_name]["Untuned"] = 1000

        t.update(1)

        # testing vgg16 source of issue
        if disable_conv2d and ("conv2d" in wkl_name):
            wkl_standalone_times[wkl_name]["Untuned"] = 0.1

        for tuned_wkl_id in tuned_wkl_ids:
            # if (
            #     tuned_wkl_id
            #     != "8b91a34899c8b5defc4583a1723bc7eac943af65594b51f0eed7654c853a75f8"
            # ):
            #     continue
            # if wkl_name == "fused_nn.contrib_conv2d_NCHWc_add_nn.relu_1056bb":
            #     print(
            #         wkl_classes[wkl_id] == wkl_classes_tuned[tuned_wkl_id],
            #         wkl_classes[wkl_id],
            #         wkl_classes_tuned[tuned_wkl_id],
            #     )
            # else:
            #     continue

            if wkl_classes[wkl_id] != wkl_classes_tuned[tuned_wkl_id]:
                # skip if we are a different workload class
                continue

            tuned_log_file = os.path.join(tuned_logfile_dir, f"{tuned_wkl_id}.json")
            new_logf = os.path.join(tuned_temp_dir, f"{wkl_id}_{tuned_wkl_id}.json")

            if not os.path.exists(tuned_log_file):
                t.update(1)
                print("Error")
                x += 1
                continue
            transfer_tune_log_file(
                full_wkl_ids[wkl_id],
                tuned_log_file,
                new_logf,
            )

            wkl_time, _ = test_workload(new_logf)

            wkl_standalone_times[wkl_name][tuned_wkl_id] = wkl_time
            x += 1
            t.update(1)

        # order the candidates by best time
        # get candidate standalone workload/tuning pairs to evaluate in the full model

        wkl_standalone_times[wkl_name] = {
            k: v
            for k, v in sorted(
                wkl_standalone_times[wkl_name].items(), key=lambda item: item[1]
            )
        }
        mapping[wkl_id] = list(wkl_standalone_times[wkl_name].keys())[0]
        if mapping[wkl_id] == "Untuned":
            mapping[wkl_id] = None

    t.close()  # close the progress bar
    end = time.time()
    timings["standalone"] = end - start
    start = time.time()

    # normally we use `with` syntax here, but we need the context to be long-lived
    tvm_context = tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    )
    tvm_context.__enter__()
    ceng = get_compile_engine()
    ceng.clear()

    # create and evaluate the best logfile
    top_mapping = mapping.copy()
    profiling_dir = "/tmp/tvmdbg"
    first_pass_best_time, std_time = evaluate_mapping_v2(
        mod,
        params,
        test_inputs,
        target_outputs,
        top_mapping,
        tuned_temp_dir,
        network_name,
        network_path,
        device_name,
        profiling=True,
        runs=10,
        profiling_dir=profiling_dir,
    )
    # raise ValueError(f"Print hey there lad, {first_pass_best_time}")
    # unlikely case where we get an error
    # fall back to the untuned time
    if first_pass_best_time is None:
        raise ValueError(f"We crashed somehow, oh dear.  Full evals:", full_evaluations)
        # raise ValueError(f"We crashed somehow, oh dear")
        # for k, v in top_mapping.items():
        #     top_mapping[k] = None

        # first_pass_best_time, std_time = evaluate_mapping(
        #     top_mapping,
        #     tuned_temp_dir,
        #     network_name,
        #     network_path,
        #     device_name,
        #     profiling=True,
        #     runs=10,
        #     profiling_dir=profiling_dir,
        # )
    print("Best time after the first pass is:", first_pass_best_time)
    print(top_mapping)
    end = time.time()
    timings["first_measure"] = end - start
    print(timings)

    prof_dir = os.path.join(profiling_dir, "_tvmdbg_device_CPU_0")
    profiling_names = get_tracing_names_2(
        prof_dir, wkl_full_params, tasks, task_weights
    )
    # get the performance of each workload in the full model
    start = time.time()
    tuned_wkl_inf_times = get_wkl_profiling_inf_time(
        model_wkl_ids,
        wkl_full_params,
        tasks,
        task_weights,
        prof_dir,
    )
    end = time.time()
    timings["profile_processing"] = end - start
    print("tuned_wkl_inf_times", tuned_wkl_inf_times)

    cache_names = get_cache_names(tasks, task_weights, wkl_full_params)
    # run full evalations to see if there are better approaches
    start = time.time()
    if full_evaluations > 1:
        best_time, mapping = full_selection_policy_1(
            mod,
            params,
            test_inputs,
            target_outputs,
            first_pass_best_time,
            tuned_wkl_inf_times,
            wkl_standalone_times,
            top_mapping,
            tuned_temp_dir,
            network_name,
            network_path,
            device_name,
            full_evaluations - 1,  # -1 because we already checked our best of stage 1
            wkl_names,
            profiling_names,
            cache_names,
            tasks,
            task_weights,
            wkl_full_params,
        )
    else:
        best_time = first_pass_best_time
        mapping = top_mapping

    end = time.time()
    timings["selection_policy"] = end - start

    # start = time.time()
    # _, _ = evaluate_mapping(
    #     mapping,
    #     tuned_temp_dir,
    #     network_file,
    #     device_name,
    #     profiling=True,
    #     profiling_dir="/tmp/tvmdbg",
    #     runs=5,
    # )

    # best_wkl_inf_times = get_wkl_profiling_inf_time(
    #     list(wkl_names.values()),
    #     wkl_full_params,
    #     tasks,
    #     task_weights,
    #     os.path.join("/tmp/tvmdbg", "_tvmdbg_device_CPU_0"),
    # )
    # end = time.time()
    # timings["final_step"] = end - start

    best_wkl_inf_times = {}
    print(best_wkl_inf_times)
    print(top_mapping)
    print(mapping)
    print(
        f"Evaluations: {full_evaluations}",
        "First pass:",
        first_pass_best_time,
        f"Second pass: {best_time}",
    )
    print("Timings:", timings)

    shutil.rmtree(tuned_temp_dir)

    ceng.clear()
    return best_time, timings, mapping, wkl_standalone_times


def profile_tt(
    mod,
    params,
    test_inputs,
    target_outputs,
    device_name: str,
    tasks: List[SearchTask],
    wkl_classes: dict,
    wkl_classes_tuned: dict,
    wkl_names: dict,
    full_wkl_ids: dict,
    network_name: str,
    network_path: os.PathLike,
    tuned_logfile_dir: os.PathLike,
    full_evaluations: Optional[int] = 10,
    task_weights: Optional[List[int]] = None,
    wkl_full_params=None,
):
    logger = logging.getLogger("transfer-tuning")
    model_wkl_ids = list(wkl_classes.keys())
    tuned_wkl_ids = list(wkl_classes_tuned.keys())

    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    wkl_standalone_times = dict()

    tuned_temp_dir = tempfile.mkdtemp()

    num_iterations = estimate_iterations(
        model_wkl_ids,
        tuned_wkl_ids,
        wkl_classes,
        wkl_classes_tuned,
    )
    t = tqdm(total=num_iterations)
    x = 0
    invalid_wkl_schedule_combos = []
    # for each workload in the new network, find the best tuned workload in the old logfile to use
    for i, wkl_id in enumerate(model_wkl_ids):
        wkl_name = wkl_names[wkl_id][0]
        wkl_standalone_times[wkl_name] = {}

        # get time for workload with no tuning
        wkl_time, _ = test_untuned_workload(tasks[i])
        wkl_standalone_times[wkl_name]["No Tuning"] = wkl_time

        t.update(1)

        for tuned_wkl_id in tuned_wkl_ids:
            if wkl_classes[wkl_id] != wkl_classes_tuned[tuned_wkl_id]:
                # skip if we are a different workload class
                continue

            tuned_log_file = os.path.join(tuned_logfile_dir, f"{tuned_wkl_id}.json")
            new_logf = os.path.join(tuned_temp_dir, f"{wkl_id}_{tuned_wkl_id}.json")

            if not os.path.exists(tuned_log_file):
                print("log file does not exist:", tuned_log_file)
                t.update(1)
                x += 1
                continue

            transfer_tune_log_file(
                full_wkl_ids[wkl_id],
                tuned_log_file,
                new_logf,
            )

            wkl_time, error_no = test_workload(new_logf)
            if error_no != 0:
                invalid_wkl_schedule_combos.append((wkl_name, tuned_wkl_id))
            # print(wkl_time)
            # exit(1)
            wkl_standalone_times[wkl_name][tuned_wkl_id] = wkl_time
            x += 1
            t.update(1)

    t.close()  # close the progress bar

    wkl_full_model_times = dict()

    # get the inference time of untuned workloads
    untuned_med_time, std_time = evaluate_mapping(
        mod,
        params,
        test_inputs,
        target_outputs,
        {},
        None,
        network_name,
        network_path,
        device_name,
        profiling=True,
        profiling_dir="/tmp/tvmdbg",
        runs=5,
    )
    untuned_wkl_inf_times = get_wkl_profiling_inf_time(
        model_wkl_ids,
        wkl_full_params,
        tasks,
        task_weights,
        os.path.join("/tmp/tvmdbg", "_tvmdbg_device_CPU_0"),
    )

    # get the time for each workload/schedule pair in the full model
    for i, wkl_id in enumerate(model_wkl_ids):
        wkl_name = wkl_names[wkl_id][0]
        wkl_full_model_times[wkl_name] = dict()

        wkl_full_model_times[wkl_name]["Untuned"] = untuned_wkl_inf_times[wkl_id]

        for tuned_wkl_id in tuned_wkl_ids:
            if wkl_classes[wkl_id] != wkl_classes_tuned[tuned_wkl_id]:
                # skip if we are a different workload class
                continue
            if (wkl_name, tuned_wkl_id) in invalid_wkl_schedule_combos:
                # skip if this workload schedule combo is invalid
                continue
            # create and evaluate the logfile
            mapping = {wkl_id: tuned_wkl_id}
            profiling_dir = "/tmp/tvmdbg"
            try:
                best_time, std_time = evaluate_mapping(
                    mapping,
                    tuned_temp_dir,
                    network_file,
                    device_name,
                    profiling=True,
                    profiling_dir=profiling_dir,
                )
            except Exception as e:
                logging.exception("Full model error")
                continue

            wkl_inf_time = get_wkl_profiling_inf_time(
                [wkl_id],
                wkl_full_params,
                tasks,
                task_weights,
                os.path.join(profiling_dir, "_tvmdbg_device_CPU_0"),
            )[wkl_id]

            wkl_full_model_times[wkl_name][tuned_wkl_id] = wkl_inf_time

    shutil.rmtree(tuned_temp_dir)

    return wkl_standalone_times, wkl_full_model_times


def rerun_workloads(wkl_id, tunings, tuning_dir):
    tuned_times = dict()
    for t in tunings:
        if t == "Untuned":
            tuned_times[t] = 1000
        else:
            logf = os.path.join(tuning_dir, f"{wkl_id}_{t}.json")
            wkl_time, _ = test_workload(logf, repeats=50, parallel=True)
            tuned_times[t] = wkl_time
    best = min(tuned_times, key=tuned_times.get)
    if best != tunings[0]:
        switch_out = 1
    else:
        switch_out = 0
    return best, switch_out


def tt_internetwork_main_2(
    mod,
    params,
    test_inputs,
    target_outputs,
    device_name: str,
    tasks: List[SearchTask],
    wkl_classes: dict,
    wkl_classes_tuned: dict,
    wkl_names: Dict[str, str],
    full_wkl_ids: dict,
    network_name: str,
    network_path: os.PathLike,
    tuned_logfile_dir: os.PathLike,
    task_weights: Optional[List[int]] = None,
    wkl_full_params=None,
    save_standalone_info=False,
    disable_conv2d=False,
):

    timings = {}
    model_wkl_ids = list(wkl_classes.keys())
    tuned_wkl_ids = list(wkl_classes_tuned.keys())

    best_log_file = "/tmp/neo.log"
    open(best_log_file, "w").close()

    tuned_temp_dir = tempfile.mkdtemp()
    mapping = dict()
    num_iterations = estimate_iterations(
        model_wkl_ids,
        tuned_wkl_ids,
        wkl_classes,
        wkl_classes_tuned,
    )
    wkl_standalone_times = dict()
    t = tqdm(total=num_iterations)
    x = 0


    start = time.time()

    switch_outs = 0
    # for each workload in the new network, find the best tuned workload in the old logfile to use
    for i, wkl_id in enumerate(model_wkl_ids):
        wkl_name = wkl_names[wkl_id]

        wkl_standalone_times[wkl_name] = {}

        # get time for workload with no tuning
        # wkl_standalone_times[wkl_name]["Untuned"] = untuned_wkl_inf_times[wkl_name]
        wkl_standalone_times[wkl_name]["Untuned"] = 1000

        t.update(1)

        # testing vgg16 source of issue
        if disable_conv2d and ("conv2d" in wkl_name):
            wkl_standalone_times[wkl_name]["Untuned"] = 0.1

        for tuned_wkl_id in tuned_wkl_ids:
            if wkl_classes[wkl_id] != wkl_classes_tuned[tuned_wkl_id]:
                # skip if we are a different workload class
                continue

            tuned_log_file = os.path.join(tuned_logfile_dir, f"{tuned_wkl_id}.json")
            new_logf = os.path.join(tuned_temp_dir, f"{wkl_id}_{tuned_wkl_id}.json")

            if not os.path.exists(tuned_log_file):
                t.update(1)
                print("Error: file does not exist", tuned_log_file)
                x += 1
                continue
            transfer_tune_log_file(
                full_wkl_ids[wkl_id],
                tuned_log_file,
                new_logf,
            )

            wkl_time, _, violation = test_workload(new_logf)

            if violation:
                wkl_standalone_times[wkl_name][tuned_wkl_id] = 1e10
            else:
                wkl_standalone_times[wkl_name][tuned_wkl_id] = wkl_time
            x += 1
            t.update(1)

        # order the candidates by best time
        # get candidate standalone workload/tuning pairs to evaluate in the full model
        wkl_standalone_times[wkl_name] = {
            k: v
            for k, v in sorted(
                wkl_standalone_times[wkl_name].items(), key=lambda item: item[1]
            )
        }

        top_k = list(wkl_standalone_times[wkl_name].keys())[0:10]
        new_top = top_k[0]
        switch_out = 0
        switch_outs += switch_out

        mapping[wkl_id] = new_top
        if mapping[wkl_id] == "Untuned":
            mapping[wkl_id] = None

    t.close()  # close the progress bar
    end = time.time()
    timings["standalone"] = end - start
    start = time.time()

    # print(wkl_standalone_times)
    # exit(1)
    # normally we use `with` syntax here, but we need the context to be long-lived
    tvm_context = tvm.transform.PassContext(
        opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    )
    tvm_context.__enter__()
    ceng = get_compile_engine()
    ceng.clear()

    # create and evaluate the best logfile
    top_mapping = mapping.copy()
    # profiling_dir = "/tmp/tvmdbg"
    first_pass_best_time, std_time = evaluate_mapping_v2(
        mod,
        params,
        test_inputs,
        target_outputs,
        top_mapping,
        tuned_temp_dir,
        network_name,
        network_path,
        device_name,
        profiling=False,
        runs=5,
        # profiling_dir=profiling_dir,
    )

    # fall back to the untuned time
    if first_pass_best_time is None:
        # raise ValueError(f"We crashed somehow, oh dear.  Full evals:")
        ...

    print("Best time after the first pass is:", first_pass_best_time)
    print(f"We had {switch_outs} switch_outs")
    print(top_mapping)
    timings["first_measure"] = end - start

    best_wkl_inf_times = {}
    print(best_wkl_inf_times)
    print(top_mapping)
    print(mapping)
    print(
        "First pass:",
        first_pass_best_time,
    )
    print("Timings:", timings)
    print(f"We had {switch_outs} switch_outs")
    shutil.rmtree(tuned_temp_dir)

    ceng.clear()
    return first_pass_best_time, timings, mapping, wkl_standalone_times
