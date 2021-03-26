import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


# 观察一个未经训练的随机智能体
state = env.reset()
for _ in range(10000):
    env.render()
    next_state, reward, done, _ =env.step(env.action_space.sample())
    # print(reward)




