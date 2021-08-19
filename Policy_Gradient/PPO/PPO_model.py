import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal


class ActorDiscrete(nn.Module):
    """
    用于离散动作空间的策略网络
    """
    def __init__(self,state_size,action_size):
        super(ActorDiscrete, self).__init__()
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
        """
        返回 action 和 action的概率
        """
        # probs for each action (2d tensor)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        ## return action for current state, and the corresponding probability
        # result_dic={"action":action.item(),"log_prob":m.log_prob(action)
        #     ,"prob":probs[:,action.item()].item()}

        return action.item(),probs[:,action.item()].item()


class ActorContinous(nn.Module):
    """
    用于连续动作空间的策略网络
    """
    def __init__(self,state_size,action_size):
        super(ActorContinous, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.mu_head = nn.Linear(128, action_size)
        self.sigma_head = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)

    def act(self,state):
        """
        返回 action 和 action 的 log prob
        """
        with torch.no_grad():
            (mu, sigma) = self.forward(state)  # 2d tensors
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.numpy()[0], action_log_prob.numpy()[0]


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
   pass


