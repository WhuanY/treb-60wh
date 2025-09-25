#!/bin/bash
python ./src/main.py --exp_id 3 --rep_layer 8 --training_strategy linear_probing --data_size 40000 > ./logs/3.log 2>&1
