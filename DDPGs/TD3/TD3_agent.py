import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from DDPGs.TD3.TD3_model import Actor,Critic
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./board/logs')


LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic


class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind:
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(
            dones).reshape(-1, 1)

    def __len__(self):
        return len(self.storage)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, env):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)

        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.env = env
        self.memory=ReplayBuffer()

    def select_action(self, state, noise=0.1):
        """Select an appropriate action from the agent policy

            Args:
                state (array): current state of environment
                noise (float): how much noise to add to acitons

            Returns:
                action (float): action clipped within action range

        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:      # 加入随机噪声
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        # clip action, 把 action 的值限制在一定范围内
        return action.clip(self.env.action_space.low, self.env.action_space.high)


    def train(self, iterations, batch_size, discount, tau, policy_noise,noise_clip, policy_freq):
        """Train and update actor and critic networks

            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
                batch_size(int): batch size to sample from replay buffer
                discount (float): discount factor
                tau (float): soft update for main networks to target networks

            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network

        """
        # 从 replay buffer 中批量随机采样，重复 N 次训练
        for it in range(iterations):
            # Sample replay buffer
            s, a, s1, r, d = self.memory.sample(batch_size)
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(s1).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            """
            # 1. 训练 Critic 网络
            """
            # # ----- 计算 critic loss ------- # #

            # select next action (with noise)
            noise = torch.FloatTensor(a).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)   # double critic, 选最小值
            target_Q = reward + (done * discount * target_Q).detach()
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # calculate loss function
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # # ------ 更新 Critic 网络参数 ------ # #
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            """
            # 2. 训练 Actor 网络  
            """
            # 延迟策略更新
            if it % policy_freq == 0:

                # # ----- 计算 actor loss ------- # #
                action_pred = self.actor(state)
                # 使用 Critic 网络中的一个（Q1）计算 Q value 的估计值
                actor_loss = -self.critic.Q1(state,action_pred ).mean()

                # # ------- 更新 Actor 网络参数 -------# #
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # # ----- soft update 更新目标网络 ------- # #
                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory,filename):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
