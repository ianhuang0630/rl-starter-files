import argparse
import time
import numpy
import torch

import utils
import utils.tasks as tasks  # script won't be executed if it's been already loaded
# Parse arguments
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--model-type", required=True,
                    help="model type: vanilla | optlib (REQUIRED)")  # added

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
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")


args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

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
                    use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent
if args.gif:
    from array2gif import write_gif

# Create a window to view the environment
env.render('human')
this_subtask=''
for episode in range(args.episodes):
    if args.gif:
        frames = []

    obs = env.reset()
    
    counter = 0
    while True:
        counter += 1
        wholefig_frame = env.render('human', title='experiment {}: {}'.format(episode+1, this_subtask))
        if args.gif:
            foo = numpy.moveaxis(env.render("rgb_array"), 2, 0)
            frames.append(foo)
            # save wholefig_frame
            im = Image.fromarray(wholefig_frame)
            im.save(args.gif+'{}_frame{}.png'.format(episode+1, counter))

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
            break

    if not done:
        print("Failed experiment number {}".format(episode+1))
    else:
        # save gif
        if args.gif:
            print("Saving gif... ", end="")
            write_gif(numpy.array(frames), args.gif+"{}.gif".format(episode+1), fps=1/args.pause)
            print("Done.")

    if env.window.closed:
        break

