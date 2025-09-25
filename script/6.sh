#!/bin/bash
python ./src/main.py --exp_id 6 --rep_layer last --training_strategy unfreeze_last_4 --data_size 40000 > ./logs/6.log 2>&1
