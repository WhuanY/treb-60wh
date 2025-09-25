#!/bin/bash
python ./src/main.py --exp_id 9 --rep_layer "8,12" --training_strategy linear_probing --data_size 5000 > ./logs/9.log 2>&1
