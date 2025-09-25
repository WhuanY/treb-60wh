#!/bin/bash
python ./src/main.py --exp_id 5 --rep_layer last --training_strategy unfreeze_last_2 --data_size 40000 > ./logs/5.log 2>&1 &
