#!/bin/bash
python ./src/main.py --exp_id 8 --rep_layer last --training_strategy linear_probing --data_size 5000 > ./logs/8.log 2>&1
