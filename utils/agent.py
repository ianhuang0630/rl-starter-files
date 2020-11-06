import torch

import utils
from model import ACModel, OpLibModel
import utils.tasks as tasks  # script won't be executed if it's been already loaded

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_type, model_dir,
                 device=None, argmax=False, num_envs=1, use_memory=False,
                 use_text=False, taskset_size=None):

        self.model_type = model_type

        if self.model_type == 'vanilla':
            obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
            self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        elif self.model_type == 'optlib':
            obs_space, self.preprocess_obss = utils.get_obss_optlib_preprocessor(obs_space)
            # TODO these need to be replaced! to visualize the procedurlaly generated ones!
            vocab_size = tasks.vocab.size
            if taskset_size is None:
                taskset_size = tasks.global_task_setup.num_unique_tasks
            self.acmodel = OpLibModel(obs_space, action_space, vocab_size, taskset_size)

        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        preds = {}
        with torch.no_grad():
            if self.acmodel.optlib:
                dist, value, switch, prob_out, prob_in = self.acmodel(preprocessed_obss)
                preds['switch'] = switch.cpu().numpy()
                preds['prob_out'] = prob_out.cpu().numpy()
                preds['prob_in'] = prob_in.cpu().numpy()
            else:
                if self.acmodel.recurrent:
                    dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
                else:
                    dist, _ = self.acmodel(preprocessed_obss)
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        preds['action'] = actions.cpu().numpy()
        return preds

    def get_action(self, obs):
        preds = self.get_actions([obs])
        return {key: preds[key][0] for key in preds}

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
