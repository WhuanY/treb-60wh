#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
python ./src/main.py --exp_id 2 --rep_layer last --training_strategy linear_probing --data_size 40000 > ./logs/2.log 2>&1
