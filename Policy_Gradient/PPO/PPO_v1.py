"""
PPO_V1: 直接使用累积奖励计算loss；无critic，只有policy网络
"""
import numpy as np
import gym
from collections import namedtuple
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from Policy_Gradient.PPO.PPO_model import ActorContinous,ActorDiscrete,Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA=0.99
LR=0.001
BATCH_SIZE=32
CLIP=0.2
UPDATE_TIME=10
max_grad_norm=0.5
Transition = namedtuple('Transition', ['state', 'action',  'prob', 'reward'])


class PPO_v1():

    def __init__(self, state_size, action_size,continuous=False):
        self.policy = ActorDiscrete(state_size, action_size).to(device)
        self.continuous = continuous
        if self.continuous:
            self.policy = ActorContinous(state_size, action_size).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(), lr=LR)
        self.trajectory=[]

    def update_policy(self,exps,i_episode):
        """
        update policy for every sampled transition groups
        called by learn() multiple times for one episode
        """
        states,actions,old_probs,f_Rewrds=exps
        # get action probs from new policy
        if self.continuous:
            (mus, sigmas) = self.policy(states)
            dists = Normal(mus, sigmas)
            new_probs = dists.log_prob(actions)
            ratios = torch.exp(new_probs - old_probs)
        else:
            new_probs = self.policy(states).gather(1, actions)
            ratios = new_probs / old_probs

        # calculate clipped surrogate function
        surr1 = ratios * f_Rewrds
        surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * f_Rewrds
        policy_loss=-torch.min(surr1,surr2).mean()

        # update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.optimizer.step()

        # self.traintime_counter+=1

    def learn(self,i_episode):
        """
        agent learn after finishing every episode.
        learn from experiences of this trajectory
        :return:
        """
        states=torch.cat([t.state for t in self.trajectory])
        actions=torch.tensor([t.action for t in self.trajectory],dtype=torch.long).view(-1,1)
        old_probs=torch.tensor([t.prob for t in self.trajectory],dtype=torch.float).view(-1,1)

        # -- calculate discount future rewards for every time step
        rewards = [t.reward for t in self.trajectory]
        fur_Rewards = []
        for i in range(len(rewards)):
            discount = [GAMMA ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            fur_Rewards.append(sum(d * f for d, f in zip(discount, f_rewards)))
        fur_Rewards=torch.tensor(fur_Rewards,dtype=torch.float).view(-1,1)

        for i in range(UPDATE_TIME):
            # -- repeat the flowing update loop for several times
            # disorganize transitions in the trajectory into sub groups
            for index_set in BatchSampler(SubsetRandomSampler(range(len(self.trajectory))), BATCH_SIZE, False):
                exps=(states[index_set],actions[index_set],old_probs[index_set],fur_Rewards[index_set])
                # -- update policy network for every sub groups
                self.update_policy(exps,i_episode)

        del self.trajectory[:]  # clear trajectory


    def train(self,env,i_episode):
        state = env.reset()
        total_reward=0
        while True:
            # self.timesetp_counter+=1
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            action, prob = self.policy.act(state)  # 离散空间取直接prob，连续空间取log prob
            next_state, reward, done, _ = env.step(action)

            # --store transition in this current trajectory
            self.trajectory.append(Transition(state,action,prob,reward))
            state=next_state
            total_reward+=reward
            if done:
                break
        # --agent learn after finish current episode, and if there is enough transitions
        if BATCH_SIZE <= len(self.trajectory):
            self.learn(i_episode)

        return total_reward





