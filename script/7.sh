#!/bin/bash
python ./src/main.py --exp_id 7 --rep_layer last --training_strategy full_fine_tuning --data_size 5000 > ./logs/7.log 2>&1
