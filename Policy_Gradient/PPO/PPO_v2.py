"""
PPO_V2: 使用优势函数计算loss；有critic网络
"""
import random
import numpy as np
import gym
from collections import namedtuple
from collections import deque
import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from Policy_Gradient.PPO.PPO_model import ActorContinous,ActorDiscrete,Critic
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./board/logs')

GAMMA=0.99
LR_a=0.001
LR_c=0.003
BATCH_SIZE=32
CLIP=0.2
BETA=0.01
UPDATE_TIME=10
max_grad_norm=0.5

Transition = namedtuple('Transition', ['state', 'action',  'prob', 'reward'])


class Memory():
    def __init__(self):
        self.trajectory=[]
        self.Transition = namedtuple('Transition', ['state', 'action', 'prob', 'reward'])

    def add(self,state,action,prob,reward):
        # state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.trajectory.append(self.Transition(state,action,prob,reward))

    def clean_buffer(self):
        del self.trajectory[:]

    def get_trajectory(self):
        states = torch.cat([t.state for t in self.trajectory])
        actions = torch.tensor([t.action for t in self.trajectory], dtype=torch.long).view(-1, 1)
        probs = torch.tensor([t.prob for t in self.trajectory], dtype=torch.float).view(-1, 1)
        rewards = [t.reward for t in self.trajectory]
        return states,actions,probs,rewards

    def __len__(self):
        return len(self.trajectory)


class PPO_v2():
    def __init__(self,state_size, action_size,continuous=False,add_entropy=True):

        self.critic = Critic(state_size)
        self.policy = ActorDiscrete(state_size, action_size).to(device)
        self.continuous = continuous
        if self.continuous:
            self.policy = ActorContinous(state_size, action_size).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_a)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_c)

        self.memory = Memory()
        self.train_step = 0
        self.add_entropy=add_entropy

    def policy_loss(self,states,actions,
                    old_probs,f_Rewrds,V):

        # get action probs from new policy and calculate the ratio
        if self.continuous:
            (mus, sigmas) = self.policy(states)
            dists = Normal(mus, sigmas)
            new_probs = dists.log_prob(actions)
            ratios = torch.exp(new_probs - old_probs)
        else:
            new_probs = self.policy(states).gather(1, actions)
            ratios = new_probs / old_probs

        # calculate advance from critic network
        advantage = (f_Rewrds - V).detach()

        # calculate clipped surrogate function
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
        policy_loss = -torch.min(surr1, surr2)

        if self.add_entropy:
            # include a regularization term,this steers new_policy towards 0.5
            # add in 1.e-10 to avoid log(0) which gives nan
            entropy= -(new_probs*torch.log(old_probs+1.e-10)+ (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
            policy_loss+=BETA*entropy

        policy_loss=torch.mean(policy_loss)

        return policy_loss

    def critic_loss(self,f_Rewrds, V):
        return F.mse_loss(f_Rewrds, V)

    def update_policy(self,exps,i_episode):
        states, actions, old_probs, f_Rewrds = exps
        V = self.critic(states)

        # -- update policy(actor) network -- #
        policy_loss = self.policy_loss(states,actions,old_probs,f_Rewrds,V)
        # self.writer.add_scalar('loss/policy_loss', policy_loss, global_step=self.train_step)
        # update parameters
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optimizer.step()

        # -- update value(critic) network -- #
        value_loss = self.critic_loss(f_Rewrds,V)
        # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.train_step)
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()

        self.train_step+=1
        writer.add_scalar('loss/policy_loss',policy_loss.item(),i_episode)
        writer.add_scalar('loss/value_loss', value_loss.item(), i_episode)
        writer.flush()

    def learn(self,i_episode):
        """
        agent learn after finishing every episode.
        learn from experiences of this trajectory
        :return:
        """
        # states=torch.cat([t.state for t in self.memory.trajectory])
        # actions=torch.tensor([t.action for t in self.memory.trajectory],dtype=torch.long).view(-1,1)
        # old_probs=torch.tensor([t.prob for t in self.memory.trajectory],dtype=torch.float).view(-1,1)
        # rewards = [t.reward for t in self.memory.trajectory]

        states, actions, old_probs, rewards = self.memory.get_trajectory()
        # -- calculate discount future rewards for every time step
        fur_Rewards = []
        for i in range(len(rewards)):
            discount = [GAMMA ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            fur_Rewards.append(sum(d * f for d, f in zip(discount, f_rewards)))
        fur_Rewards=torch.tensor(fur_Rewards,dtype=torch.float).view(-1,1)

        for i in range(UPDATE_TIME):
            # -- repeat the flowing update loop for several times
            # disorganize transitions in the trajectory into sub groups
            for index_set in BatchSampler(SubsetRandomSampler(range(self.memory.__len__())), BATCH_SIZE, False):
                exps=(states[index_set],actions[index_set],old_probs[index_set],fur_Rewards[index_set])
                # -- update policy network for every sub groups
                self.update_policy(exps,i_episode)

        self.memory.clean_buffer()

    def train(self,env,i_episode):
        state = env.reset()
        total_reward=0
        while True:
            # self.timesetp_counter+=1
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            action,prob = self.policy.act(state) # 离散空间取直接prob，连续空间取log prob
            next_state, reward, done, _ = env.step(action)
            # --store transition in this current trajectory
            self.memory.add(state,action,prob,reward)
            state=next_state
            total_reward+=reward
            if done:
                break
        # --agent learn after finish current episode, and if there is enough transitions
        if BATCH_SIZE <= self.memory.__len__():
            self.learn(i_episode)

        return total_reward




