from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch
import arrow
import os
from DDPGs.TD3.TD3_new import TD3

RESUME= True
SAVE_MODEL_EVERY = 5
load_checkpoint_patch=["models/checkpoint/actor_10.pth","models/checkpoint/critic_10.pth"]


def output_scores(start_time,i_episode,scores_deque,score):
    print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'
          .format(i_episode, np.mean(scores_deque), score), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}\t Running time til now :{}'
              .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
    if np.mean(scores_deque) >= 300:
        print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}\t Total running time :{}'
                .format(i_episode, np.mean(scores_deque),arrow.now()-start_time))
        return True

    return False


def watch_smart_agent(agent,filename_actor,filename_crtic):
    agent.actor.load_state_dict(torch.load(filename_actor))
    agent.critic.load_state_dict(torch.load(filename_crtic))
    state = env.reset()
    for t in range(1000):
        action = agent.select_action(state)
        print(action)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


def watch_random_agent():

    for _ in range(5):
        env.reset()
        while True:
            env.render()
            next_state, reward, done, _ =env.step(env.action_space.sample())
            if done:
                break

    env.close()


def plot_scores(scores,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    plt.show()


def save_check_point(agent,i_episode):
    # setting the check point for training
    checkpoint_actor = {
        "net": agent.actor.state_dict(),
        'optimizer': agent.actor_optimizer.state_dict(),
        "epoch": i_episode
    }
    checkpoint_critic = {
        "net": agent.critic.state_dict(),
        "optimizer": agent.critic_optimizer.state_dict(),
        "epoch": i_episode
    }
    if not os.path.isdir("models/checkpoint"):
        os.mkdir("models/checkpoint")
    torch.save(checkpoint_actor, 'models/checkpoint/actor_%s.pth' % (str(i_episode)))
    torch.save(checkpoint_critic, 'models/checkpoint/critic_%s.pth' % (str(i_episode)))


def load_check_point(agent):
    "load saved checkpoints to resume training"
    checkpoint_actor = torch.load(load_checkpoint_patch[0])  # 加载断点
    checkpoint_critic = torch.load(load_checkpoint_patch[1])

    agent.actor.load_state_dict(checkpoint_actor['net'])  # 加载模型可学习参数
    agent.critic.load_state_dict(checkpoint_critic['net'])

    agent.actor_optimizer.load_state_dict(checkpoint_actor['optimizer'])  # 加载优化器参数
    agent.critic_optimizer.load_state_dict(checkpoint_critic['optimizer'])  # 加载优化器参数

    start_epoch = checkpoint_actor['epoch']  # 设置开始的epoch
    return start_epoch


def train_td3(env,agent,n_episodes):
    start_epoch = 1

    if RESUME:    # 加载 check point 中保存的模型参数继续训练
        start_epoch=load_check_point(agent)

    scores_deque = deque(maxlen=100)
    scores = []
    start_time = arrow.now()
    for i_episode in range(start_epoch, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        time_step = 0

        # loop over time steps
        while True:
            # 智能体选择动作（根据当前策略）
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.save_exp(state, action, next_state, reward, done)
            if agent.mode==1:
                agent.train(time_step)
            time_step += 1
            state = next_state
            total_reward += reward
            if done:
                break

        # recording scores
        scores.append([i_episode,total_reward])
        scores_deque.append(total_reward)
        finished = output_scores(start_time, i_episode, scores_deque, total_reward)
        if finished:
            agent.save('models', 'TD3_v2')
            break

        if i_episode% SAVE_MODEL_EVERY ==0:
            save_check_point(agent, i_episode)
            # 同时保存 scores，存为 scv 文件
            scores_df=pd.DataFrame(data=scores,columns=['episode','score'])
            scores_df.to_csv('scores_saved.csv',index=False)

        if agent.mode==0:
            agent.train(time_step)

    return scores


if __name__=="__main__":
    env = gym.make('BipedalWalker-v3')
    env.seed(10)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent_0 = TD3(state_dim,action_dim,max_action,env,0)      # mode=0:update per episode
    agent_1 = TD3(state_dim, action_dim, max_action, env, 1)  # mode=1: update per time step
    # scores=train_td3(env,agent_1,1000)

    # 观察未经训练的随机智能体
    #watch_random_agent()
    watch_smart_agent(agent_0,"models/TD3_actor.pth","models/TD3_critic.pth")


