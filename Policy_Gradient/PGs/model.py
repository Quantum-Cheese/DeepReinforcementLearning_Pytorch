import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Policy(nn.Module):
    "single Policy network for Reinforce and PG"

    def __init__(self,state_size,action_size):
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, action_size)

    def forward(self, x):
        """
        Build a network that maps state -> action probs.
        """

        out=F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out),dim=1)

        return out

    def act(self,state):
        # probs for each action (2d tensor)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        # return action for current state, and the corresponding probability

        result_dic={"action":action.item(),"log_prob":m.log_prob(action)
            ,"prob":probs[:,action.item()].item()}
        return result_dic


class Actor(nn.Module):
    """Policy netwrok for PPO_R"""
    "Actor_Critic model for PPO_A"

    def __init__(self,state_size,action_size):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(0)
        self.fc1 = nn.Linear(state_size, 128)
        # self.fc2 = nn.Linear(64,128)
        self.fc2= nn.Linear(128, action_size)

    def forward(self, x):
        """
        Build a network that maps state -> action probs.
        """

        x=F.relu(self.fc1(x))
        out = F.softmax(self.fc2(x),dim=1)
        return out

    def act(self,state):
        # probs for each action (2d tensor)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        # return action for current state, and the corresponding probability

        result_dic={"action":action.item(),"log_prob":m.log_prob(action)
            ,"prob":probs[:,action.item()].item()}
        return result_dic


class Critic(nn.Module):
    " Actor_Critic model for PPO"

    def __init__(self,state_size):
        super(Critic, self).__init__()
        self.fc1=nn.Linear(state_size,128)
        # self.fc2=nn.Linear(64,128)
        self.fc2=nn.Linear(128,1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        state_value = self.fc2(x)
        return state_value


if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = np.array([-0.04456399, 0.04653909, 0.01326909, -0.02099827])
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    policy=Policy(state_size=4,action_size=2).to(device)
    action,log_prob=policy.act(state)
    print(action,log_prob)



