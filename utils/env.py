import gym
import gym_minigrid
"""
gym_minigrid wrapper with symbols
"""
class OptLibEnv(gym.Env):
    optlib = True  # just an identifier to tell others that it is for optlib
    def __init__(self, env, task):
        self.env = env
        self.task = task
        self.symbSeq = task.symbs
        self.symbidx = 0

    def step(self, actions, switches):
        """
        actions include whether or not to switch
        to next symbol
        """
        # symbidx
        obs, reward, done, info = self.env.step(actions)
        # check what the action says -- should we increment the symbidx?
        if switches.item():
            # move the index
            self.symbidx += 1
            self.symbidx = min(self.symbidx, len(self.symbSeq)-1)

        obs = self.append_symb_task(obs)
        return obs, reward, done, info

    @property
    def observation_space(self):
        # add now the observation of task and symbol ids
        obs_space = self.env.observation_space
        return obs_space

    @property
    def action_space(self):
        return self.env.action_space

    def append_symb_task(self, obs):
        obs['curr_symbol'] = self.task.encode_symbol(self.symbSeq[self.symbidx])
        if self.symbidx + 1 == len(self.symbSeq):
            obs['next_symbol'] = self.task.encode_symbol(self.symbSeq[self.symbidx])
        else:
            obs['next_symbol'] = self.task.encode_symbol(self.symbSeq[self.symbidx+1])
        obs['task'] = [1]  # TODO keep as a single 1 for now
        return obs

    def reset(self):
        """ resetting symbidx before resetting env.
        """
        self.symbidx = 0
        # TODO: return the same observations tructure (with symbs and task id info)
        obs = self.env.reset()
        obs = self.append_symb_task(obs)
        return obs

def make_env(env_key, seed=None, optlib=False, task=None):
    env = gym.make(env_key)
    env.seed(seed)
    if optlib:
        assert task is not None
        env = OptLibEnv(env, task)
    return env
