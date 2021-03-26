import torch
import arrow
import gym
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from Policy_Gradient.PPO.PPO_v2 import PPO_v2
from Policy_Gradient.PPO.PPO_v1 import PPO_v1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def output_scores(start_time,i_episode,scores_deque,score,solve_limit):
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
          .format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\t Running time til now :{}'
              .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
    if np.mean(scores_deque) >= solve_limit:
        print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}\t Total running time :{}'
                .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
        return True

    return False


def plot_scores(scores,filename):
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    plt.show()


def get_env_prop(env_name, continuous):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    return env,state_dim, action_dim


def train_agent_for_env(env_name,continuous,n_episode,model_file,solve_limit):
    """
    continuous: 动作空间是否连续（True/False)
    model_file: 训练好的模型的保存路径
    solve_limit : 环境 solve 的标准，score 阈值
    """
    env, state_dim, action_dim =  get_env_prop(env_name,continuous)

    agent = PPO_v1(state_dim,action_dim,continuous)
    scores_deque = deque(maxlen=100)
    scores = []

    start_time = arrow.now()
    for i_episode in range(1, n_episode + 1):
        total_reward = agent.train(env,i_episode)
        # record scores(total rewards) per episode
        scores_deque.append(total_reward)
        scores.append(total_reward)
        solved = output_scores(start_time, i_episode, scores_deque, total_reward,solve_limit)
        if solved:
            torch.save(agent.policy.state_dict(), model_file)
            break

    return agent, scores


def watch_random_agent(env_name,continuous):
    env, state_dim, action_dim = get_env_prop(env_name, continuous)
    for _ in range(5):
        env.reset()
        while True:
            env.render()
            next_state, reward, done, _ =env.step(env.action_space.sample())
            if done:
                break

    env.close()


def watch_smart_agent(env_name,continuous,model_name,n_episode):
    env,state_dim, action_dim = get_env_prop(env_name,continuous)
    agent=PPO_v1(state_dim,action_dim,continuous)
    agent.policy.load_state_dict(torch.load(model_name))

    scores =[]
    for i_episode in range(1, n_episode + 1):
        rewards = []
        state = env.reset()
        while True:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action, _ = agent.policy.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores.append(sum(rewards))
    return scores


if __name__=="__main__":
    """train PPO agent in CartPole (discrete action space)"""
    # agent_cartPole,scores_1 =  train_agent_for_env('CartPole-v0',False,2000,
    #                                                'models/cartPole_ppo-v1_1.pth',195)
    # plot_scores(scores_1,'cartPole_ppo-v1_1.png')

    # 观察未经训练的随机智能体
    # watch_random_agent('CartPole-v0',False)
    # 测试训练好的智能体
    # test_scores=watch_smart_agent('CartPole-v0',False,'models/PPO_new.pth',100)
    # plot_scores(test_scores,"PPO_cartPole_test.png")

    """train PPO agent in MountainCarContinuous (continuous action space)"""
    agent_mCar, scores_2 = train_agent_for_env('MountainCarContinuous-v0', True, 2000,
                                               'models/mCar_ppo-v1.pth',95)
    plot_scores(scores_2, 'mCar_ppo-v1_1.png')








