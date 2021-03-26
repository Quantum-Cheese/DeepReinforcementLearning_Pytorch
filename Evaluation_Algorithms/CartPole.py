"""
在 CartPole 环境中测试多种算法智能体的表现，并对比奖励曲线图
测试算法：1.PPO
        2.DDPG/TD3
        3.DQN
        4.A3C/A2C
"""

import torch
import gym
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt


def plot_scores(scores,file_name,multi_time=False):
    "绘制多次训练多条曲线"
    if multi_time:
        x=np.arange(1, len(scores[0]) + 1)
        for n in range(len(scores)):
            rolling_mean = pd.Series(scores[n]).rolling(100).mean()
            plt.plot(x,rolling_mean,label="trial_"+str(n+1))
    else:
        x = np.arange(1, len(scores) + 1)
        rolling_mean = pd.Series(scores).rolling(100).mean()
        plt.plot(x, rolling_mean)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def plot_diff_agent(scores_2d,file_name):
    " 绘制多种不同agent的训练曲线：多曲线图"
    for name,scores in scores_2d:
        x = np.arange(1, len(scores) + 1)
        rolling_mean = pd.Series(scores).rolling(100).mean()
        plt.plot(x, rolling_mean,label=name)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(file_name)
    plt.show()

