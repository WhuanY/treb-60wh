#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

python ./src/main.py --exp_id 1 --rep_layer last --training_strategy full_fine_tuning --data_size 40000 > ./logs/1.log 2>&1
