import argparse
import time
import numpy
import torch

import utils
import utils.tasks as tasks  # script won't be executed if it's been already loaded
import os
# Parse arguments
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", required=True,
                    help="model type: vanilla | optlib (REQUIRED)")  # added


parser.add_argument('--procs', type=int, default=4,
                    help = "number of processes used during training")
parser.add_argument('--procedural', action="store_true", default=False)
parser.add_argument("--env", default=None,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--task-id", type=str, default=None)
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--save-frames", action="store_true", default=False,
                    help="save frames")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

args = parser.parse_args()


# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

if args.procedural:
    # create the tasks
    task_envs, task_list = tasks.get_procedural_taskenvs(args.procs, 5, max_length=5, seed=args.seed)
    task_set = tasks.TaskSet(task_list, tasks.vocab)
    task_setup = tasks.TaskSetup([task_list, None])
    envs = {}
    unique_tasks = []
    for idx, task in enumerate(task_list):
        # envs.append(utils.make_env(args.env, args.seed + 10000 * i))
        envs[task.id] = [utils.make_env(instance,
                                        args.seed + 10000 * idx,
                                        optlib = args.model_type == 'optlib',
                                        task=task) for instance in task_envs[idx]]

        # this will be a list
        unique_tasks.append(task.id)
    assert len(envs) == len(unique_tasks) # Tasks x processes


if args.model_type == 'optlib' or args.task_id is not None:
    assert args.task_id is not None, '--task-id cannot be None if using optlib'
    task = tasks.get_task(args.task_id)
    env_name = task.env
else:
    task = None
    env_name = args.env
# Load environment
env = utils.make_env(env_name, args.seed,
                     optlib=args.model_type == 'optlib',
                     task=task)

for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, args.model_type,
                    model_dir, device=device, argmax=args.argmax,
                    use_memory=args.memory, use_text=args.text, taskset_size=len(unique_tasks) if args.procedural else None)
print("Agent loaded\n")

# Run the agent
if args.gif:
    from array2gif import write_gif

# Create a window to view the environment
env.render('human')
this_subtask=''

# creating folders for frames
if args.task_id is not None:
    dir_path = os.path.join('viz', args.model, args.task_id)
else:
    dir_path = os.path.join('viz', args.model, args.env)

for episode in range(args.episodes):
    if args.gif:
        frames = []

    obs = env.reset()
    this_subtask = tasks.get_subtask_id(task, obs['curr_symbol'])

    counter = 0
    while True:
        counter += 1

        # generating new frame to save.
        label = 'experiment {}: {}'.format(episode+1, this_subtask)
        print(label)
        wholefig_frame = env.render('human', title=label)

        if args.gif:
            foo = numpy.moveaxis(env.render("rgb_array"), 2, 0)
            frames.append(foo)

        if args.save_frames:
            # save wholefig_frame
            im = Image.fromarray(wholefig_frame)
            # folder is viz/model_name/environment-name/frame_number.png
            subdir_path = os.path.join(dir_path, 'episode{}'.format(episode+1))
            file_path = os.path.join(subdir_path, 'frame{}.png'.format(counter))
            utils.create_folders_if_necessary(file_path)
            im.save(file_path)

        preds = agent.get_action(obs)
        if args.model_type == 'vanilla':
            obs, reward, done, _ = env.step(preds['action'])
        elif args.model_type == 'optlib':
            obs, reward, done, _ = env.step(preds['action'],
                                            switches=preds['switch'])
            # announce the current symbol
            this_subtask = tasks.get_subtask_id(task, obs['curr_symbol'])
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            # final save
            label = 'experiment {}: {}'.format(episode+1, this_subtask)
            print("DONE! " + label)
            wholefig_frame = env.render('human', title=label)

            if args.gif:
                foo = numpy.moveaxis(env.render("rgb_array"), 2, 0)
                frames.append(foo)

            if args.save_frames:
                # save wholefig_frame
                im = Image.fromarray(wholefig_frame)
                subdir_path = os.path.join(dir_path, 'episode{}'.format(episode+1))
                file_path = os.path.join(subdir_path, 'frame{}.png'.format(counter+1))
                utils.create_folders_if_necessary(file_path)
                im.save(file_path)
            break

    # save gif
    if args.gif:
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), args.gif+"{}.gif".format(episode+1), fps=1/args.pause)
        print("Done.")

    if env.window.closed:
        break

