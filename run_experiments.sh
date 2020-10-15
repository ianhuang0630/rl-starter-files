#!/bin/bash

# first argument = model directory for saving
# second argument = model type. can be optlib or vanilla

# pretraining
python3 -m scripts.train_optlib --algo a2c --model "$1" --save-interval 10 --frames 800000 --procs 4 --model-type "$2" 

# transfer 
python3 -m scripts.train_optlib --algo a2c --model "$1"_transfer --save-interval 10 --frames 800000 --procs 4 --model-type "$2" --transfer --pretrained-model "$1"



