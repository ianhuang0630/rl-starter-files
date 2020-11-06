#!/bin/bash

# first argument = model directory for saving
# 2nd argument = number of episodes

## first base task (to-room-with-goal)
python -m scripts.visualize_optlib --model "$1" --task-id to-room-with-goal --model-type optlib --episodes "$2" --save-frames --procedural

## second base task (to-goal)
python -m scripts.visualize_optlib --model "$1" --task-id to-goal --model-type optlib --episodes "$2" --save-frames --procedural

## third base task (to-goal)
python -m scripts.visualize_optlib --model "$1" --task-id 4-room --model-type optlib --episodes "$2" --save-frames --procedural

## fourth base task (unlock)
python -m scripts.visualize_optlib --model "$1" --task-id unlock --model-type optlib --episodes "$2" --save-frames --procedural

## first transfer task (door-key)
python -m scripts.visualize_optlib --model "$1" --task-id door-key --model-type optlib --episodes "$2" --save-frames --procedural

## second transfer task (simplje-crossing)
python -m scripts.visualize_optlib --model "$1" --task-id simple-crossing --model-type optlib --episodes "$2" --save-frames --procedural


