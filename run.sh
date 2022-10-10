#!/usr/bin/env bash

# Run transfer-tuning for a specified model, for a specified number of full evaluations
python3 src/scripts/tt_single_model_neo.py \
    --network_file models/vanilla/resnet18.onnx `#Chosen model to optimize`  \
    --device_name  xeon_cpu \
    --original_network_file models/vanilla/resnet50.onnx `#Model used for tuning` \
    --split_log_file_dir data/processed/split_logs `#Pre-tuned workloads (we select the workloads from original_network_file)` \
    --full_evaluations 10 `#Number of full evaluations to run`

# Run the full set of experiments
OUTPUT_FILE=data/results/tt_multi_models_results_main.json

python3 src/scripts/tt_multi_models.py \
    --split_log_file_dir data/processed/split_logs \
    --network_path models/vanilla \
    --device_name  xeon_cpu \
    --output_file $OUTPUT_FILE

# Generate the initial set of plots

# Tune Ansor for the time we used
python3 src/scripts/autoschedule_models.py \
    --network_path models/vanilla \
    --device_name xeon_cpu \
    --output_dir data/raw/final_results \
    --tt_file $OUTPUT_FILE
