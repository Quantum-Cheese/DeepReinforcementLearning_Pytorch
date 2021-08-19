import argparse
import os
import arrow
import torch
import gym
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from DQNs.DQN_cnn.dqn_agent import Agent_dqn
from DQNs.DQN_cnn import atari_wappers


def plot_scores(scores,filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)



def train_agent(env,agent,n_episode,eps_decay,gamma,update_every):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    start_time = arrow.now()
    for i_episode in range(1, n_episode + 1):
        state = env.reset()
        print(state.shape)
        score = 0
        episode_loss=[]
        while True:
            # # check the memory usage of system, clean replay buffer if too high
            # if (sys_mem.used / sys_mem.total) >= 0.03:
            #     agent.memory.clean_buffer()
            #     print('Buffer cleaned on episode {}'.format(i_episode))
            # get action
            action = agent.act(state,i_episode,eps_decay)
            # interact with env (one step)
            next_state, reward, done, _ = env.step(action)
            # train the agent
            sarsd = (state, action, reward, next_state,done)
            loss = agent.step(sarsd,gamma,update_every)
            # update status
            state = next_state
            score += reward
            # break the loop if current episode is over
            if done:
                break
            if loss is not None:
                episode_loss.append(loss)

        # update rewards and scores every episode
        scores_window.append(score)
        scores.append(score)

        # print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode, np.mean(episode_loss),
        #                                                                np.mean(scores_window)), end="")
        #
        # if i_episode > 25:
        #     print('Replay Buffer size: {}'.format(agent.memory.__len__()))
        #     print('Memory used: ',sys_mem.used)
        #     print('Memory used rate: ',sys_mem.used/sys_mem.total)

        if i_episode % 100 == 0:
            print('\rEpisode {}\t Loss {} \t Average Score: {:.2f}'.format(i_episode, np.mean(episode_loss),
                                                                           np.mean(scores_window)))
            print('\rRunning time till now :{}\n'.format(arrow.now() - start_time))


    print("Training finished, total running time:{}. \n Model saved.".format(arrow.now()-start_time))

    return scores





if __name__ =="__main__":
    env = atari_wappers.make_env("SpaceInvaders-v0")
    state_size, action_size = env.observation_space.shape, env.action_space.n
    dqn_agent = Agent_dqn(state_size,action_size)
    train_agent(env,dqn_agent,1,0.98,0.995,5)
