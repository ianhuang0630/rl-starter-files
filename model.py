import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

from torch.distributions.bernoulli import Bernoulli

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# TODO: a new type of model, OpLibModel(ACModel)
#####

class OpLibModel(nn.Module, torch_ac.OpLibModelBase):
    """
    Maintains a library of options for every function within a dsl

    A single option is composed of 3 sets of parameters:
    - theta_in: to parametrize an estimator of a state falling within the input
                set of the current option
    - theta_out: to parametrize an estimator of a state falling within the
                output set of the estimator
    - theta_pol: the policy parameters
    """
    def __init__(self, obs_space, action_space, vocab_embedding_size=5, task_embedding_size=1):
        super().__init__()
        # image-related transforms
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        ########################################################

        # TODO: intialized with a one-to-on mapping between function and option
        # random initialization of theta_in, theta_term, theta_pol
        # TODO: temperature is set to 0 for every option
        self.output_dim = action_space.n
        self.vocab_embedding_size = vocab_embedding_size
        self.task_embedding_size = task_embedding_size

        # below are the probability distributions over the state space,
        # conditioned on symbols
        self.prob_in = nn.Sequential(
            nn.Linear(self.image_embedding_size + self.vocab_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.prob_out = nn.Sequential(
            nn.Linear(self.image_embedding_size + self.vocab_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # below are actor and critic
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size + self.vocab_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size + self.task_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )


    def forward(self, obs):

        emb = self.get_embedding(obs)
        emb_curr_symb = torch.cat((emb, obs.curr_symbol), dim=1)
        emb_next_symb = torch.cat((emb, obs.next_symbol), dim=1)
        emb_task = torch.cat((emb, obs.task), dim=1)

        prob_out = self.prob_out(emb_curr_symb)
        bdist = Bernoulli(prob_out)
        switch = bdist.sample()

        prob_in_curr = self.prob_in(emb_curr_symb)
        prob_in_next = self.prob_in(emb_next_symb)
        prob_in = (1-switch)*prob_in_curr + switch*prob_in_next
        value = self.critic(emb_task)
        value = value.squeeze(1)

        policy_curr = self.actor(emb_curr_symb)
        policy_next = self.actor(emb_next_symb)
        logits = (1-switch)*policy_curr + switch*policy_next
        dist = Categorical(logits=F.log_softmax(logits, dim=1))

        return dist, value, switch, prob_out, prob_in

    def get_embedding(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        emb = x.reshape(x.shape[0], -1)
        return emb

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
