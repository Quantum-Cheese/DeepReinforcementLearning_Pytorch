import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gym
import arrow
import torch
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from dqn_per import Agent_dqn
import atari_wappers
# from DQNs.DQN_PER.dqn_per import Agent_dqn
# from DQNs.DQN_PER import atari_wappers


def train_agent(agent,state_size,n_episodes ):
    scores_window = deque(maxlen=100)  # last 100 scores
    scores , eps_lst = [],[]

    start_time = arrow.now()
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        while True:
            action,epsilon = agent.act(state,i_episode)
            next_state, reward, done, _ = env.step(action)

            ## add sample and train agent
            sarsd = (state, action, reward, next_state, done)
            agent.step(sarsd)

            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps_lst.append(epsilon)

        print('\rEpisode {} \t Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\t  Average Score: {:.2f}'.format(i_episode,np.mean(scores_window)))
            print('\rRunning time:{}\n'.format(arrow.now() - start_time))
        # if np.mean(scores_window) >= 195.0:
        #     print('\nEnvironment solved in {:d} episodes! \t Average Score: {:.2f}'.format(i_episode - 100,
        #                                                                                    np.mean(scores_window)))
        #     # torch.save(agent.qnetwork_local.state_dict(), model_file)
        #     print('\nTotal running time:{}'.format(arrow.now() - start_time))
        #     break

    return scores,eps_lst


def plot_curves(data,plot_name,filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(data)), data)
    plt.ylabel(plot_name)
    plt.xlabel('Episode #')
    plt.savefig(filename)


if __name__=="__main__":
    env = atari_wappers.make_env("SpaceInvaders-v0")
    state_size, action_size = env.observation_space.shape, env.action_space.n

    cnn_agent = Agent_dqn(state_size,action_size,'CNN','True','nonlinear')
    train_scores, _ = train_agent(cnn_agent, state_size, 2500)
    plot_curves(train_scores, 'Scores', 'Plots/cnn_per.png')

    # env = gym.make('CartPole-v0')
    # env.seed(0)
    # state_size, action_size = env.observation_space.shape[0], env.action_space.n
    # mlp_agent = Agent_dqn(state_size, action_size,'MLP','True','nonlinear')
    # train_scores,eps_lst = train_agent(mlp_agent,state_size,2500)
    # plot_curves(train_scores,'Scores','Plots/train_exp-3.png')
    # if mlp_agent.eps_decay:
    #     plot_curves(eps_lst,'Epsilon', 'Plots/epsilon_exp-3.png')
