from plot_fig5 import get_main_results, get_untuned_times, get_fig5

x = get_main_results("/workspace/data/results/tt_multi_models/")
print(x)


y = get_untuned_times("/workspace/data/raw/chocolate/")
print(y)

get_fig5("/workspace/data/results/tt_multi_models/")
