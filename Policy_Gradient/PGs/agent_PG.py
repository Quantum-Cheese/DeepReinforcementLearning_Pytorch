import numpy as np
import gym
from collections import deque
import torch
import torch.optim as optim
from CartPole.Policy_Gradient.model import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=1.0
LR=0.001


class Agent_PG():

    def __init__(self, state_size, action_size,type):
        self.policy=Policy(state_size,action_size).to(device)
        self.optimizer=optim.Adam(self.policy.parameters(), lr=LR)
        self.type=type

    def reinforce_loss(self,log_probs,rewards):
        "------根据 Reinforce 算法计算的损失函数---------"
        # calculate discount rewards
        discounts=[GAMMA**i for i in range(len(rewards))]
        R=sum([g*r for g,r in zip(discounts,rewards)])

        loss_arr=[]
        for log_prob in log_probs:
            loss_arr.append(-log_prob * R)

        policy_loss=torch.cat(loss_arr).sum()  # 把n个1d tensor 组成的list 拼接成一个完整的 tensor（1d,size:n）
        # print(policy_loss)
        return policy_loss

    def pg_loss(self,log_probs,rewards):
        """----
        Reinforce 的改进版本：
        1.Credit Assignment：对每个 a(t) 计算未来累积折扣回报 R
        2.对每个t的回报R进行 batch normalization
        ------"""
        # calculate the (discounted) future rewards
        furRewards_dis = []
        for i in range(len(rewards)):
            discount = [GAMMA ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            furRewards_dis.append(sum(d * f for d, f in zip(discount, f_rewards)))
        # print(furRewards_dis)

        # -- Normalize reward
        mean = np.mean(furRewards_dis)
        std = np.std(furRewards_dis) + 1.0e-10
        rewards_normalized = (furRewards_dis - mean) / std

        # -- calculate policy loss
        loss_arr = []
        for i in range(len(rewards_normalized)):
            loss_arr.append(-log_probs[i]*rewards_normalized[i])
        # print(loss_arr)

        policy_loss = torch.cat(loss_arr).sum()
        # print(policy_loss,"----------\n")

        return policy_loss

    def train(self,env):
        state = env.reset()
        log_probs = []
        rewards = []
        # --- collect log probs and rewards for a single trajectory
        while True:
            # convert state to tensor
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # 升维 1d->2d
            result_dic = self.policy.act(state)
            next_state, reward, done, _ = env.step(result_dic['action'])
            log_probs.append(result_dic['log_prob'])
            rewards.append(reward)
            state = next_state
            if done:
                break
        total_reward = sum(rewards)

        # --- update policy after one completed trajectory
        # calculate loss
        loss = self.reinforce_loss(log_probs, rewards)
        if self.type=="reinforce":
            loss = self.reinforce_loss(log_probs, rewards)
        elif self.type=="pg":
            loss = self.pg_loss(log_probs, rewards)

        # backprop the loss to update policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return total_reward


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    agent=Agent_PG(state_size=4,action_size=2,type='pg')
    n_episode=2000

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1,n_episode+1):
        Reward=agent.train(env)

        scores_deque.append(Reward)
        scores.append(Reward)
        if i_episode % 100 == 0:
            print('Episode {}\t Average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))