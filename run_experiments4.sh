#!/bin/bash

# parsing arguments
optlib=false # whether to train optlib
vanilla=false # whether to train vanilla
generatetask=false # whether to generate tasks
visualizetasks=false
transfer=false
memory=false
randomcurriculum=false

# default parameters in general
numprocs=4
# default parameters for task generation
taskloc=""
excludeloc=""
taskmaxbf=2
taskseqperlen=5
taskmaxlen=5
taskroomsize=6

numframesperproc=32

frames_base="--frames 6400000" # default: 6400000
frames_transfer="--frames 1600000" # default: 1600000

# colon indicates a parameter that needs an argument
while getopts m:n:ovstl:r:p:e:x:fyc:d flag
do
    case "${flag}" in
        m) modelname=${OPTARG};;
        n) vanillaname=${OPTARG};;
        o) optlib=true;;
        v) vanilla=true;;
        s) visualizetasks=true;;
        t) generatetask=true;;
        f) transfer=true;;
        l) taskloc=${OPTARG};;
        r) transfertaskloc=${OPTARG};;
        x) excludeloc=${OPTARG};; # location of tasks to exclude from task generation
        p) numprocs=${OPTARG};; # number of processes
        e) seed=${OPTARG};;
        y) memory=true;;
        c) recurrence=${OPTARG};;
        d) randomcurriculum=true;;
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
    if [ -d "task_envs/$taskloc" ]; then
        echo "Task already exists at $taskloc. Delete first, then rerun."
        exit 1
    fi

    command="python3 -m scripts.generate_environments --procs $numprocs --max-length $taskmaxlen --seq-per-length $taskseqperlen --max-branch-factor $taskmaxbf --room-size $taskroomsize --save-dir $taskloc"

    if [ "$excludeloc" ]; then
        echo "Setting exclude directory"
        excludeadjunct="--exclude-dir $excludeloc --min-length $taskmaxlen"
        command="$command $excludeadjunct"
    fi

    if [ "$seed" ]; then
        echo "Setting seed to $seed"
        seedadjunct="--seed $seed"
        command="$command $seedadjunct"
    fi
    echo "$command"
    eval "$command" || exit 1
fi

# (optional) visualization of task
if [ "$visualizetasks" = true ]; then
    echo "Running task visualization"
fi

# training
command="python3 -m scripts.train_optlib4 --algo a2c --save-interval 10  --procs $numprocs --frames-per-proc $numframesperproc"

if [ "$memory" = true ]; then
    if [ -n "$recurrence" ]; then
        command="$command --recurrence $recurrence"
    else
        echo "ERROR: recurrence number not provided through -c"
        exit 1
    fi
fi

if [ "$randomcurriculum" = true ]; then
    command="$command --random-curriculum"
fi

# command="python3 -m scripts.train_optlib4 --algo a2c --save-interval 10  --procs $numprocs"
if [ ! "$taskloc" ]; then
    echo "ERROR: no location provided to save tasks environments" 
    exit 1
fi
if [ ! -d "task_envs/$taskloc" ]; then
    echo "ERROR: Taskloc is empty. Create tasks in this folder first."
    exit 1
fi

command="$command --task-loc $taskloc"
# if eventually going to transfer, the command needs to be appended with information
if [ "$transfer" = true ]; then
    if [ -n "$transfertaskloc" ]; then
        if [ ! -d "task_envs/$transfertaskloc" ]; then
            echo "ERROR: the task location you specified for task tasks does not exist"
        fi
    else
        echo "ERROR: you didn't specify a task location for the transfer tasks"
    fi
    echo "adding transfer task location"
    command="$command --transfer-task-loc $transfertaskloc"
fi

# NOTE: using the above arguments, it should be able to register all the tasks (transfer and base).
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
    optlibcommand="$optlibcommand $frames_base"
    echo "Training oplib model"
    echo "$optlibcommand"
    eval "$optlibcommand"

    echo "Copying task information to folder"
    cp -rf "task_envs/$taskloc" "storage/$modelname/" || exit 1
    echo "Done"

fi

# transfer learning on transfer set --transfer-task-loc $transfertaskloc --transfer
if [ "$transfer" = true ] && [ "$optlib" = true ] ; then
    # check if a model name is given
    if [ -n "$modelname" ]; then
        echo "transfer-$modelname"
    else
        echo "ERROR: You didn't specify a model name through -m."
        exit 1
    fi
    # check tha tthe model is already trained.
    if [ -d "storage/$modelname" ]; then
        echo "pretrained model found. MOVING ON."
    else
        echo "ERROR: pretrained model not found"
        exit 1
    fi

    if [ -d "storage/transfer-$modelname" ]; then
        echo "duplicate folder found. Remove, then rerun."
        exit 1
    else
        echo "NO duplicates found. MOVING ON."
    fi
    optlibcommand="$command --model-type optlib --model transfer-$modelname --pretrained-model $modelname --transfer"
    optlibcommand="$optlibcommand $frames_transfer"
    echo "TRANSFER-Training oplib model"
    echo "$optlibcommand"
    eval "$optlibcommand"

    echo "Copying task information to folder"
    cp -rf "task_envs/$transfertaskloc" "storage/transfer-$modelname/" || exit 1
    echo "Done"
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
    vanillacommand="$vanillacommand $frames_base"
    echo "Training vanilla model"
    echo "$vanillacommand"
    eval "$vanillacommand"

    echo "Copying task information to folder"
    cp -rf "task_envs/$taskloc" "storage/$vanillaname/" || exit 1
    echo "Done"
fi

# transfer learning on transfer set --transfer-task-loc $transfertaskloc --transfer
if [ "$transfer" = true ] && [ "$vanilla" = true ]; then
    # check if a model name is given
    if [ -n "$vanillaname" ]; then
        echo "transfer-$vanillaname"
    else
        echo "ERROR: You didn't specify a vanilla name through -n."
        exit 1
    fi
    # check tha tthe model is already trained.
    if [ -d "storage/$vanillaname" ]; then
        echo "pretrained model found. MOVING ON."
    else
        echo "ERROR: pretrained model not found"
        exit 1
    fi

    if [ -d "storage/transfer-$vanillaname" ]; then
        echo "duplicate folder found. Remove, then rerun."
        exit 1
    else
        echo "NO duplicates found. MOVING ON."
    fi
    vanillacommand="$command --model-type vanilla --model transfer-$vanillaname --pretrained-model $vanillaname --transfer"
    vanillacommand="$vanillacommand $frames_transfer"
    echo "TRANSFER-Training vanilla model"
    echo "$vanillacommand"
    eval "$vanillacommand"

    # copying task information
    echo "Copying task information to folder
    cp -rf "task_envs/$transfertaskloc" "storage/transfer-$vanillaname/" || exit 1
    echo "Done"
fi



# Step 4: visualize the model's end predictions for the base and transfer datasets
# Step 5: visualize the vanilla's end predicvtions for the base and transfer datasets
