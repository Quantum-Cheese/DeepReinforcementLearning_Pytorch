import torch
import arrow
import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


if __name__=="__main__":
    env,state_dim,action_dim = get_env_prop("CartPole-v0",False)




