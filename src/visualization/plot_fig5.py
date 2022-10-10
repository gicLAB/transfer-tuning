#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd

# set matplotlib sizes
import matplotlib as mpl

mpl.rcParams["hatch.linewidth"] = 0.6  # previous pdf hatch linewidth
mpl.rcParams["hatch.linewidth"] = 6.0  # previous svg hatch linewidth

nice_fonts = {
    "text.usetex": False,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 28,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 30,
    "legend.title_fontsize": 40,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
}
mpl.rcParams.update(nice_fonts)

def get_main_results(path: os.PathLike):
    results = dict()
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            with open(os.path.join(path, filename)) as f:
                d = json.load(f)
                assert len(d.keys()) == 1
                d = d[list(d.keys())[0]]["0"]
                v = {"tt_time": d["tt_time"], "search_time": d["search_time"]}
                results[filename[:-5]] = v
    return results


def get_untuned_times(path: os.PathLike):
    with open(os.path.join(path, "tuning_info.json")) as f:
        d = json.load(f)
    untuned_times = dict()
    for k, v in d.items():
        untuned_times[k] = v["untuned_time"]
    return untuned_times


def get_ansor_times(path: os.PathLike):
    with open(os.path.join(path, "tuning_info.json")) as f:
        d = json.load(f)
    tuned_times = dict()
    for k, v in d.items():
        tuned_times[k] = v["tuned_time"]
    return tuned_times


def get_ansor_times(path: os.PathLike, tt_results):
    ansor_times = dict()
    for filename in os.listdir(path):
        if not filename.endswith(".json"):
            continue
        model = filename[:-5]
        ansor_times[model] = {}

        with open(os.path.join(path, filename)) as f:
            d = json.load(f)
        tt_result = tt_results[model]

        # find speedup ansor gets given same search time
        for search_time, inf_time in d.items():
            search_time = float(search_time)
            if search_time >= tt_result["search_time"]:
                ansor_times[model]["tuned_time"] = inf_time
                break
        # find search time ansor requires to get same speedup
        for search_time, inf_time in d.items():
            if inf_time <= tt_result["tt_time"]:
                ansor_times[model]["search_time"] = float(search_time)
                break

             # ensures that we get a search time even if Ansor never actually beats us
            ansor_times[model]["search_time"] = float(search_time)

    return ansor_times

def get_fig5(result_path: os.PathLike):
    tt_results = get_main_results(result_path)
    untuned_times = get_untuned_times("/workspace/data/raw/chocolate/")
    ansor_compare = get_ansor_times("/workspace/data/results/tt_stop_points/", tt_results)
    # calculate speedups
    # speedups = dict()
    # all_speedups = dict()
    # ansor_diffs = []
    # potential = dict()

    # for k, v in tt_old_times.items():
    #     speedups[k] = untuned_times[k] / v

    #     tt = untuned_times[k] / v
    #     ansor = untuned_times[k] / ansor_0_evals_new[k]
    #     ansor_old = untuned_times[k] / ansor_0_evals[k]
    #     all_speedups[titles[k]] = [tt, ansor]
    #     ansor_diffs.append([ansor, ansor_old])
    #     potential[k] = (1 - (untuned_times[k] / v)) / (
    #         1 - (untuned_times[k] / ansor_tuned_times[k])
    #     )
    skip = ["bert_128", "mobilebert_128"]
    all_speedups = dict()
    speedups = dict()
    diffs = dict()
    for k, v in tt_results.items():
        if k in skip:
            continue
        v = v["tt_time"]
        speedups[k] = untuned_times[k] / v

        tt = untuned_times[k] / v
        ansor = untuned_times[k] / ansor_compare[k]["tuned_time"]
        all_speedups[titles[k]] = [tt, ansor]
        diffs[k] = (tt - 1) / (ansor - 1)

    # plot speedups figure
    plot_speedup_fig(all_speedups, diffs)

    # plot search time figure
    plot_search_time_fig(tt_results, ansor_compare)



def plot_speedup_fig(all_speedups, diffs):
    # plot speedups fig
    df = pd.DataFrame(all_speedups).T

    df = df.rename(columns={0: "Transfer-Tuning", 1: "Ansor"})
    df["model"] = df.index
    df["model"] = pd.Categorical(df["model"], list(titles.values()))
    df = df.sort_values("model")

    cm_div = ["blue", "yellow"]
    cm_div[0] = plt.get_cmap("viridis")(0.3)
    cm_div[1] = "lightgreen"

    sc = 30
    fig, ax = plt.subplots(figsize=((1 / 2) * sc * 1.61803, (2 / 3) ** 4 * sc))

    bars = df.plot(
        kind="bar", ax=ax, color=cm_div, rot=45, edgecolor="black", linewidth=3, width=0.7
    )

    ymax = 1.5
    ax.set_ylim([0.8, ymax])


    ax.set_ylabel("Speedup (×)")
    ax.yaxis.grid()
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("3")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    plt.show()
    fig.savefig("/workspace/fig_5a.png", bbox_inches="tight")

# Nicely formatted names
titles = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "alexnet": "AlexNet",
    "vgg16": "VGG-16",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3": "MobileNetV3",
    "efficentnetb0": "EfficentNetB0",
    "efficentnetb4": "EfficentNetB4",
    "efficentnetb7": "EfficentNetB7",
    "googlenet": "GoogLeNet",
    "mnasnet0_5": "MnasNet0.5",
    "mnasnet1_0": "MnasNet1.0",
    "bert-base-uncased-seq_class-128": "BERT (128)",
    "bert-base-uncased-seq_class-256": "BERT",
    "mobilebert-base-uncased-seq_class-128": "MobileBERT (128)",
    "mobilebert-base-uncased-seq_class-256": "MobileBERT",
}


def plot_search_time_fig(tt_results, ansor_results):
    all_speedups = dict()
    skip = ["bert_128", "mobilebert_128"]

    # exit(1)
    for k, _ in ansor_results.items():
        if k in skip:
            continue
        tt = tt_results[k]["search_time"] / 60
        try:
            ansor = ansor_results[k]["search_time"] / 60
        except:
            print("error")
            print(k)
            print(ansor_results[k])
            exit(1)
        all_speedups[titles[k]] = [tt, ansor]

    df = pd.DataFrame(all_speedups).T

    df = df.rename(columns={0: "Transfer-Tuning", 1: "Ansor"})
    df["model"] = df.index
    df["model"] = pd.Categorical(df["model"], list(titles.values()))
    df = df.sort_values("model")

    cm_div = ["blue", "yellow"]
    cm_div[0] = plt.get_cmap("viridis")(0.3)
    cm_div[1] = "lightgreen"


    sc = 30
    fig, ax = plt.subplots(figsize=((1 / 2) * sc * 1.61803, (2 / 3) ** 4 * sc))

    bars = df.plot(
        kind="bar", ax=ax, color=cm_div, rot=45, edgecolor="black", linewidth=3, width=0.7
    )

    ymax = 32
    ax.set_ylim([0, ymax])

    ax.set_ylabel("Search time (mins)")
    ax.yaxis.grid()
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("3")
    ax.legend(ncol=1, loc="upper left")
    plt.show()
    fig.savefig("/workspace/fig_5b.png", bbox_inches="tight")
