"""
Used to create environments and subtask sequences
"""

import os
import argparse
import json
import pickle
import utils.tasks as tasks

# step 1: take in arguments
parser = argparse.ArgumentParser()
parser.add_argument("--procs", type=int, default=4,
                    help="number of processes (default: 16), number of instantiations for each subtask sequence.")
parser.add_argument("--min-length", type=int, default=1,
                    help="minimum length of task sequences")
parser.add_argument("--max-length", type=int, default=5,
                    help="maximum length of task sequences")
parser.add_argument("--seq-per-length", type=int, default=5,
                    help="number of subtask sequences to be drawn at every length.")
parser.add_argument("--max-branch-factor", type=int, default=2,
                    help="maxmimum number of rooms that can be generated within a single accesesible cluster.")
parser.add_argument("--room-size", type=int, default=6,
                    help="the size of the rooms")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--save-dir", type=str, required=True,
                    help="the directory where the environments and data are going to be saved")

parser.add_argument("--exclude-dir", type=str, default=None,
                    help="the directory for exclusion")


# TODO exclusion?
args = parser.parse_args()

if args.exclude_dir is not None:
    exclude_dirpath = os.path.join('task_envs', args.exclude_dir)
    assert os.path.exists(exclude_dirpath), "Invalid --exclude-dir"
    with open(os.path.join(exclude_dirpath, 'meta.json'), 'r') as f:
        exclude_meta = json.load(f)
        exclude_list = []
        for length_tasks in exclude_meta['task_sequences']:
            for task_seq in length_tasks:
                exclude_list.append(task_seq)

# step 2: create the graph
# step 3: create environments
env_list, task_list = tasks.get_procedural_taskenvs(args.procs, args.seq_per_length,
                                                    min_length=args.min_length,
                                                    max_length=args.max_length,
                                                    max_cluster_size=args.max_branch_factor,
                                                    seed=args.seed, exclude=None if args.exclude_dir is None else exclude_list)

# NOTE: task_list doesn't have a pointer to the environment ; that's in the env_list

assert len(env_list) == len(task_list)
assert all([len(envs) == args.procs for envs in env_list])

# creating saveable dictionary (taskname -> (task, [envs] ) )
taskenvs = {}
for task, envs in zip(task_list, env_list):
    assert task.id not in taskenvs, "{} is not a unique task id".format(task.id)
    taskenvs[task.id] = (task, envs)

# creating metadata dictionary
counter = 0
task_sequence_unsqueezed = []
task_id_unsqueezed = []

for i in range(args.max_length - args.min_length + 1):
    row_sequence = []
    row_id = []
    for j in range(args.seq_per_length):
        row_sequence.append([symb.id for symb in task_list[counter].symbol_seq])
        row_id.append(task_list[counter].id)
        counter += 1
    task_sequence_unsqueezed.append(row_sequence)
    task_id_unsqueezed.append(row_id)

meta = {'subtask_vocabulary': [s.id for s in tasks.vocab.symbols],
        'task_sequences': task_sequence_unsqueezed,
        'task_ids': task_id_unsqueezed,
        'procs': args.procs,
        'max_length': args.max_length,
        'seq_per_length': args.seq_per_length,
        'max_branch_factor': args.max_branch_factor,
        'room_size': args.room_size,
        'seed': args.seed}

# preparing the save
save_dir_path = os.path.join('task_envs', args.save_dir)
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

meta_path = os.path.join(save_dir_path, 'meta.json')
taskenvs_path = os.path.join(save_dir_path, 'taskenv.pkl')

# save the environment, the graph, and input parameters, the vocab
with open(meta_path, 'w') as f:
    json.dump(meta, f)

with open(taskenvs_path, 'wb') as f:
    pickle.dump(taskenvs, f)
