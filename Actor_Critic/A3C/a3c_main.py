import gym
import numpy as np
import matplotlib.pyplot as plt
from Actor_Critic.A3C.agent_a3c import A3C


def get_env_prop(env_name, continuous):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    return env,state_dim, action_dim


def train_a3c(env_name,continuous):
    env,state_size,action_size = get_env_prop(env_name,continuous)
    agent = A3C(env,continuous,state_size,action_size)
    scores = agent.train_worker()
    return scores


def train_agent_for_env(env_name,continuous):
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    if continuous:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    agent = A3C(env, continuous,state_dim,action_dim)
    scores = agent.train_worker()

    return agent,scores


def plot_scores(scores,filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # env = gym.make("Pendulum-v0")
    # train_scores = train_a3c(env,True)

    # train A3C on discrete env : CartPole
    scores_cartPole = train_agent_for_env("CartPole-v0",False)
    plot_scores(scores_cartPole,"cartPole_trainPlot.png")

    # train A3C on continuous env : continuous
    # a3c_mCar = train_agent_for_env("MountainCarContinuous-v0", True)

