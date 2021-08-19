import gym
import arrow
import torch
import numpy as np
from collections import deque
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from DQNs.DDQN.ddqn_v3 import AgentV3


def dqn(agent,model_file,n_episodes=2000, max_t=1000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        beta_start=0.4):
    """Deep Q-Learning.


    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    beta=beta_start

    start_time=arrow.now()
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        episode_loss=[]
        for t in range(max_t):
            # 在当前状态下获取要采取的 action
            action = agent.act(state, eps)
            # 与环境交互获取 （s',r,done）
            next_state, reward, done, _ = env.step(action)
            # 构建 sarsa 序列，传给智能体
            loss=agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_loss.append(loss)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # beta = beta/beta_incre if beta<beta_end else beta_end # update beta (<=1)
        # beta=min((beta+beta_incre),beta_end)

        print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode,np.mean(episode_loss), np.mean(scores_window)),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode,np.mean(episode_loss), np.mean(scores_window)))
            print('\rRunning time:{}\n'.format(arrow.now()-start_time))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes! \t Average Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_file)
            print('\nTotal running time:{}'.format(arrow.now() - start_time))
            break
    return scores


def watch_agent(agent):

    state = env.reset()
    for j in range(500):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


def watch_random_agent():

    for _ in range(3):
        env.reset()
        while True:
            env.render()
            next_state, reward, done, _ =env.step(env.action_space.sample())
            if done:
                break

    env.close()


def trained_agent_test(filename,episode_num=500,max_t=1000,eps=0.01):
    """
    :param filename:
    :param episode_num:
    :param max_t:
    :param eps:
    :return:
    """
    # agent = Agent(state_size=8, action_size=4, seed=0)
    agent_v3 = AgentV3(state_size=8, action_size=4, seed=0)
    agent_v3.qnetwork_local.load_state_dict(torch.load(filename))

    watch_agent(agent_v3)

    scores=[]
    scores_window = deque(maxlen=100)
    start_time=arrow.now()
    for i_episode in range(episode_num):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # 直接采用贪婪策略
            action = agent_v3.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        scores_window.append(score)
        print('\rEpisode {}\t  Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t  Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rRunning time:{}\n'.format(arrow.now()-start_time))
    return scores


def plot_scores(scores,filename):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(np.arange(len(scores_1)), scores_1)
    ax.plot(np.arange(len(scores)), scores)
    # rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)


if __name__=="__main__":
    env = gym.make('LunarLander-v2')
    env.seed(0)

    # 训练 ddqn agent 并获取平均累计奖励
    agent_v3 = AgentV3(state_size=8, action_size=4, seed=0)
    print("\n\nTraining ddqn agent:\n-------------------------------------------------------------\n")
    train_scores = dqn(agent_v3,'dueling_model.pth')
    # plot_scores(train_scores,'images/dueling-ddqn_training.png')

    # 观察未经训练的随机智能体
    #watch_random_agent()
    # 用训练好的智能体跑分并绘制奖励曲线
    # test_scores=trained_agent_test('models/dueling_model.pth')
    # plot_scores(test_scores,'images/dueling-ddqn_testing.png')











