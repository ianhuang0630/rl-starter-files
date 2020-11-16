import os
import pickle
import json
import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys
from copy import deepcopy

import utils
from model import ACModel, OpLibModel, OpLibModel

# for multitask option learning
import utils.tasks as tasks  # script won't be executed if it's been already loaded
import numpy as np

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--model-type", required=True,
                    help="model type: vanilla | optlib (REQUIRED)")  # added
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--pretrained-model", default=None,
                    help="name of the model that's pretrained")
parser.add_argument("--task-loc", required=True,
                    help="name of the folder under task_envs/ that has all the task and environment information")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

parser.add_argument("--transfer-task-loc", default=None,
                    help="folder for the transfer")

parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

parser.add_argument("--transfer", action="store_true", default=False,
                    help="switch to transfer tasks, starting from the model parameters pointed to by `--model`.")

args = parser.parse_args()

if args.transfer:
    assert args.pretrained_model is not None, "--model must be specified if you are transferring to testing tasks."

args.mem = args.recurrence > 1
# Set run dir date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
# default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}" # 
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

if args.transfer:
    pretrained_model_dir = utils.get_model_dir(args.pretrained_model)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
# tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

num_gpus = torch.cuda.device_count()
if num_gpus:
    device = torch.device("cuda:{}".format(num_gpus-1))
else:
    device = torch.device("cpu")

txt_logger.info(f"Device: {device}\n")

txt_logger.info("Environments loaded\n")

# Load training status
try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

if args.transfer:
    pretrained_status = utils.get_status(pretrained_model_dir)


# Load environments for different tasks
envs = {}  # envs will become a dictionary, where each entry is a list of environment instantiations.
# environments across different tasks, and random initializations within the same task

assert os.path.exists(os.path.join('task_envs', args.task_loc))
taskenv_path = os.path.join('task_envs', args.task_loc)
with open(os.path.join(taskenv_path, 'meta.json'), 'r') as f:
    meta = json.load(f)
with open(os.path.join(taskenv_path, 'taskenv.pkl'), 'rb') as f:
    task_env = pickle.load(f)

# processing the base task set
base_task_envs, base_task_list = [], []
for key in task_env:
    taskenv_tup = task_env[key]
    assert isinstance(taskenv_tup[0], tasks.Task)
    assert isinstance(taskenv_tup[1], list)
    base_task_list.append(taskenv_tup[0])
    base_task_envs.append(taskenv_tup[1])

base_task_set = tasks.TaskSet(base_task_list, tasks.vocab)

# processing the transfer dataset
transfer_task_set, transfer_task_envs, transfer_task_list = None, None, None
if args.transfer_task_loc is not None:
    transfer_taskenv_path = os.path.join('task_envs', args.transfer_task_loc)
    with open(os.path.join(transfer_taskenv_path, 'meta.json'), 'r') as f:
        transfer_meta = json.load(f)
    with open(os.path.join(transfer_taskenv_path, 'taskenv.pkl'), 'rb') as f:
        transfer_task_env = pickle.load(f)

    transfer_task_envs, transfer_task_list = [], []
    for key in transfer_task_env:
        taskenv_tup = transfer_task_env[key]
        assert isinstance(taskenv_tup[0], tasks.Task)
        # TODO: edit ID?
        modified_task = deepcopy(taskenv_tup[0])
        modified_task.name = 'transfer-' + modified_task.id  # to avoid id collision problems down the line
        assert isinstance(taskenv_tup[1], list)
        transfer_task_list.append(modified_task)
        transfer_task_envs.append(taskenv_tup[1])

    transfer_task_set = tasks.TaskSet(transfer_task_list, tasks.vocab)

task_setup = tasks.TaskSetup([base_task_list, transfer_task_list])
# decide task_set and task_envs based on transfer/or not
task_set = transfer_task_set if args.transfer else base_task_set
task_list = transfer_task_list if args.transfer else base_task_list
task_envs = transfer_task_envs if args.transfer else base_task_envs

for idx, task in enumerate(task_list):
    # envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    envs[task.id] = [utils.make_env(instance,
                                    optlib = args.model_type == 'optlib',
                                    task=task) for instance in task_envs[idx]]

# Load observations preprocessor
# NOTE: assuming that all environments have the same observation space
# NOTE: assuming that all environments have the same action space
firstkey = list(envs.keys())[0]
if args.model_type == 'optlib':
    obs_space, preprocess_obss = utils.get_obss_optlib_preprocessor(envs[firstkey][0].observation_space)
elif args.model_type == 'vanilla':
    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[firstkey][0].observation_space)
else:
    raise ValueError('Model type invalid')

if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])  # NOTE: this is a different vocab than the symbols, remove?
txt_logger.info("Observations preprocessor loaded")

# Load model
if args.model_type == 'vanilla':
    optlibmodel = ACModel(obs_space, envs[firstkey][0].action_space, args.mem, args.text)
elif args.model_type == 'optlib':
    vocab_size = tasks.vocab.size
    taskset_size = task_setup.num_unique_tasks
    optlibmodel = OpLibModel(obs_space, envs[firstkey][0].action_space, vocab_size, taskset_size)
else:
    raise ValueError('Model type invalid')

if args.transfer:
    assert "model_state" in pretrained_status, "pretrained model should have recorded model state"
    optlibmodel.load_state_dict(pretrained_status["model_state"])

else:
    if "model_state" in status:
        optlibmodel.load_state_dict(status["model_state"])

optlibmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(optlibmodel))

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()


# iterating through a sequence of different tasks, with symbols as input into models.
txt_logger.info("Training on {} tasks".format('TRANSFER' if args.transfer else 'BASE'))

# NOTE: shuffle tasks?
# NOTE: perhaps what we should pretrain all the policy networks first.
# NOTE: making the boundaries soft?

task2algo = {} # creating dictionary mapping each task to an algorithm instantiation .
for idx, task in enumerate(task_set):
    txt_logger.info('Loading optimizing algorithm for task {}'.format(task))

    assert isinstance(task, tasks.Task), \
        "something went wrong, task needs to be of type Task"
    task_specific_envs = envs[task.id]
    # instantiate algo based on the environment of the current task
    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(task_specific_envs, optlibmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(task_specific_envs, optlibmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])

    txt_logger.info("Optimizer loaded\n")

    # saving into dictionary.
    task2algo[task.id] = algo

    txt_logger.info('TASK: {}\n'.format(task))

tasksampler = tasks.CurriculumTaskSampler(task_list, int(np.ceil(args.frames/algo.num_frames)))
# loading the task symbols.
asdf = 0 # for debugging. this number should eventually match 
while num_frames < args.frames: # a2c gives 5 frames a time by default
    # Update model parameters
    update_start_time = time.time()
    asdf += 1

    # sample task from the task distribution
    # sample_task_id = np.random.choice(list(task2algo.keys()))
    sample_task_id = tasksampler.next_sample()
    algo = task2algo[sample_task_id]

    # experiences should also include switch decisions.
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)  # updates parameters every 5 frames

    #### all logging
    logs = {**logs1, **logs2}
    update_end_time = time.time()
    num_frames += logs["num_frames"]
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["Task", "update", "frames", "FPS", "duration"]
        data = [sample_task_id, update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        # heading switch statistics
        total_switches_per_process = np.sum(logs['switches'], axis=0, dtype=np.int).tolist()

        beta_value = None
        if beta_value is None:
            infoline = "T {} | U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | C {}".format(*data, total_switches_per_process)
        else:
            header += ['beta']
            data += [beta_value]
            infoline = "T {} | U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | B {} | C {}".format(*data, total_switches_per_process)

        txt_logger.info(infoline)

        # tack on before final csv save.
        header += ['proc{}_total_switches'.format(i+1) for i in range(len(total_switches_per_process))]
        data += total_switches_per_process

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()
        if status["num_frames"] == 0 and num_frames == logs["num_frames"]:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        #for field, value in zip(header, data):
            #tb_writer.add_scalar(field, value, num_frames)

    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                    "model_state": optlibmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")

print('groundtruth: {} calculated: {}'.format(asdf, int(np.ceil(args.frames/algo.num_frames))))
