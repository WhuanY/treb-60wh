#!/bin/bash

python ./src/main.py --exp_id 10 --rep_layer last --training_strategy full_fine_tuning --data_size 40000 > ./logs/10.log 2>&1
