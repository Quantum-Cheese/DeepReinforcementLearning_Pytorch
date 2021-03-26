import gym
import numpy as np
from collections import deque


def hill_climbing(env,policy,n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []         # 用于存储各 episode 的得分（总奖励）
    best_R = -np.Inf
    best_w = policy.w

    for i_episode in range(1, n_episodes + 1):
        rewards = [] # 每个episode 重置奖励队列
        state = env.reset()
        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward) # 把当前 时间步的奖励加入 rewards 队列
            if done:
                break
        # 设定折扣率
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        # 计算当前episode的折扣累计总奖励
        R = sum([a * b for a, b in zip(discounts, rewards)])

        scores_deque.append(sum(rewards)) # 把当前episode的累计奖励（无折扣）加入 scores 队列
        scores.append(sum(rewards))

        # ------- 参数搜索 ----- #
        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2) # 缩小搜索范围（下限为 0.001）
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2) # 扩大搜索范围（上限为2）
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)
        # --------------------- #

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            policy.w = best_w
            break

    return scores