#!/bin/bash
python ./src/main.py --exp_id 4 --rep_layer "8,12" --training_strategy linear_probing --data_size 40000 > ./logs/4.log 2>&1 &
