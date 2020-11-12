#!/bin/bash

# parsing arguments
optlib=false # whether to train optlib
vanilla=false # whether to train vanilla
generatetask=false # whether to generate tasks
visualizetasks=false

# default parameters in general
numprocs=4
# default parameters for task generation
taskloc=""
taskmaxbf=2
taskseqperlen=5
taskmaxlen=5
taskroomsize=6


# colon indicates a parameter that needs an argument
while getopts m:n:ovstl:p:e: flag
do
    case "${flag}" in
        m) modelname=${OPTARG};;
        n) vanillaname=${OPTARG};;
        o) optlib=true;;
        v) vanilla=true;;
        s) visualizetasks=true;;
        t) generatetask=true;;
        l) taskloc=${OPTARG};;
        p) numprocs=${OPTARG};; # number of processes
        e) seed=${OPTARG};;
    esac
done

# generating tasks
if [ "$generatetask" = true ]; then
    echo "generating tasks"
    # check that there's a task location specified
    if [ ! "$taskloc" ]; then
        echo "ERROR: no location provided to save tasks environments"
        exit 1;
    fi

    # cleaning $taskloc if it already exists
    echo "removing duplicate task directory"
    command="python3 -m scripts.generate_environments --procs $numprocs --max-length $taskmaxlen --seq-per-length $taskseqperlen --max-branch-factor $taskmaxbf --room-size $taskroomsize --save-dir $taskloc"

    if [ ! "$seed" ]; then
        echo "$command"
        eval "$command" || exit 1
    else
        seedadjunct="--seed $seed"
        echo "Setting seed to $seed"
        echo "$command $seedadjunct"
        eval "$command $seedadjunct" || exit 1
    fi
fi

# (optional) visualization of task
if [ "$visualizetasks" = true ]; then
    echo "Running task visualization"
fi

# training # TODO: adding parameters to load task
command="python3 -m scripts.train_optlib4 --algo a2c --save-interval 10 --frames 6400000 --procs $numprocs"

if [ ! "$taskloc" ]; then
    echo "ERROR: no location provided to save tasks environments" 
    exit 1
fi
if [ ! -d "task_envs/$taskloc" ]; then
    echo "ERROR: Taskloc is empty. Create tasks in this folder first."
    exit 1
fi

command="$command --task-loc $taskloc"

if [ "$optlib" = true ]; then
    # check if a model name is given
    if [ -n "$modelname" ]; then
        echo "$modelname"
    else
        echo "ERROR: You didn't specify a model name through -m."
        exit 1
    fi
    # checking for duplicates
    if [ -d "storage/$modelname" ]; then
        echo "duplicate folder found. Remove, then rerun."
        exit 1
    else
        echo "NO duplicates found. MOVING ON."
    fi
    optlibcommand="$command --model-type optlib --model $modelname"
    echo "Training oplib model"
    echo "$optlibcommand"
    eval "$optlibcommand"
fi

if [ "$vanilla" = true ]; then
    if [ -n "$vanillaname" ]; then
        echo "$vanillaname"
    else
        echo "ERROR: You didn't specify a vanilla name through -n."
        exit 1
    fi
    # checking for duplicates
    if [ -d "storage/$vanillaname" ]; then
        echo "ERROR: duplicate folder found. Remove, then rerun."
        exit 1
    else
        echo "NO duplicates found. MOVING ON."
    fi
    vanillacommand="$command --model-type vanilla --model $vanillaname"
    echo "Training vanilla model"
    echo "$vanillacommand"
    eval "$vanillacommand"
fi

# Step 4: visualize the model's end predictions for the base and transfer datasets
# Step 5: visualize the vanilla's end predicvtions for the base and transfer datasets
