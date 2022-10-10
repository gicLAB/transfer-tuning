#!/usr/bin/env bash

python3 src/scripts/generate_model_set.py \
    --model_set chocolate

python3 autoschedule_models.py \
    --model_set_dir models/chocolate \
    --ntrials 100 \
    --device_name xeon_cpu
