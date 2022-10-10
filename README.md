Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation
==============================

Code artifact for our PACT 2022 paper.  Find the [arXiv version here](https://arxiv.org/abs/2201.05587).


```
@inproceedings{gibson_transfer_tuning_2022,
  author = {Gibson, Perry and Cano, Jos{\'e}},
  title = {{Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation}},
  booktitle = {2022 31st {{International Conference}} on {{Parallel Architectures}} and {{Compilation Techniques}} ({{PACT}})},
  year = {2022},
  location = {Chicago, IL},
  numpages = {12},
  publisher = {ACM},
  address = {New York, NY, USA}
}
```


Guide
------------

This project works best in a Docker container, for which we provide a basic template for.

You can build and run the container with:

``` sh
docker build -t tt_artifact:latest -f docker/Dockerfile.main_cpu .
docker run \
    -v ${PWD}/transfer-tuning/:/workspace \
    -ti tt_artifact:latest bash -i
```

This will install the required dependencies, build TVM, and ensure most environment variables are set correctly.

Details on running the system:

``` sh
# Export models to TVM format
python3 src/scripts/generate_model_set.py \
    --model_set chocolate

# Auto-schedule models using Ansor
python3 src/scripts/autoschedule_models.py \
   --model_path models/chocolate \
   --ntrials 20000 \
   --device_name xeon_cpu \
   --output_dir data/raw/chocolate

# use the Ansor internal tool to extract the top performing auto-schedules
# (done internally anyway, but this reduces parse time)
python3 src/scripts/distil_logfiles.py \
    --log_file_dir data/raw/chocolate

# save relevant task info for each model (e.g. kernel names)
python3 src/scripts/generate_task_info.py \
    --network_dir models/chocolate \
    --device_name xeon_cpu

# split logfiles into individual workloads
python3 src/scripts/split_logfiles.py \
    --log_file_dir data/raw/chocolate \
    --network_dir models/chocolate \
    --output_dir data/processed/split_logs/chocolate

# run TT for a single model (optional)
# python3 src/scripts/tt_single_model_pact.py \
#    --model_name mobilenetv2 \
#    --tt_model_name efficentnetb4 \
#    --split_log_file_dir data/processed/split_logs/chocolate \
#    --model_path models/chocolate \
#    --device_name xeon_cpu

# run TT with all of our models
python3 src/scripts/tt_multi_models_pact.py \
    --split_log_file_dir data/processed/split_logs/chocolate/ \
    --model models/chocolate \
    --device_name xeon_cpu

# Compare how well Ansor performs given the same search time
# as well as go beyond see how much time is required to match our time
# this stage will take a while as we need to take all of the time
# required for TT, plus time to wait for Ansor to beat TT
python3 src/scripts/autoschedule_models.py \
     --model_path models/chocolate \
     --output_dir data/raw/chocolate_tt_ansor \
     --ntrials 8000 \
     --device_name xeon_cpu \
     --tt_path /workspace/data/results/tt_multi_models \
     --minute_check

# Plot the results
python3 src/visualization/visualize.py
```

The results will vary depending on the hardware platform you run on, however you should observe the overall trend that transfer-tuning can provide speedups to models, while providing said speedups in less time than Ansor.

Of course, tuning Ansor for more time will eventually beat transfer-tuning.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docker             <- Dockerfiles and instructions.
    │   ├── Dockerfile.main_cpu
    │   └── README.md
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tvm                <- Source code for TVM - with small modifications.
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
