import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from MountCar_continuous.cross_entropy_method.agent_cem import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cem(agent,n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
    """PyTorch implementation of a cross-entropy method.

    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    n_elite = int(pop_size * elite_frac)

    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma * np.random.randn(agent.get_weights_dim())

    for i_iteration in range(1, n_iterations + 1):
        weights_pop = [best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)

        reward = agent.evaluate(best_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        torch.save(agent.state_dict(), 'checkpoint.pth')

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration - 100,
                                                                                           np.mean(scores_deque)))
            break
    return scores



def watch_trained_agent(agent):
    # load the weights from file
    agent.load_state_dict(torch.load('checkpoint.pth'))

    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    while True:
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = agent(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break

    env.close()


if __name__=="__main__":
    env = gym.make('MountainCarContinuous-v0')
    env.seed(101)
    np.random.seed(101)
    agent = Agent(env).to(device)

    # --- train and plot scores --- #
    scores = cem(agent)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # --- watch a pre-trained agent --- #
    watch_trained_agent(agent)


