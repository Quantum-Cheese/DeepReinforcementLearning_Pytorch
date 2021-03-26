import gym
import numpy as np
from collections import deque
import os
import torch
# from skimage import io
from DQNs.DQN_cnn.dqn_agent import Agent_dqn
from DQNs.DQN_cnn import atari_wappers


# def save_films(state,step):
#     if not os.path.exists('./image'):
#         os.makedirs('./image')
#     img_name = './image/pic-%d.jpg' % step
#     io.imsave(img_name, state)


def random_play():
    for step in range(5000):
        env.render()
        action = 1
        state, reward, done, info = env.step(action)

        if step % 100 == 0:
            print(state.shape)
            # print(state)
            save_films(state, step)

        if reward > 0:
            print(reward, step)
            save_films(state, step)

        if done:
            print('dead in %d steps' % step)
            break


def random_test(env):
    socres = []
    scores_window = deque(maxlen=100)

    for i_episode in range(100):
        state = env.reset()
        score = 0
        while True:
            action = np.random.choice(env.action_space.n,1)[0]
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        socres.append(score)
        scores_window.append(score)

        if i_episode % 10 == 0:
            print('Episode {},\t Average score : {} '.format(i_episode, np.mean(scores_window)))


def trained_agent_test(env,agent):
    socres = []
    scores_window = deque(maxlen=100)

    for i_episode in range(5000):
        state = env.reset()
        score = 0

        while True:
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        socres.append(score)
        scores_window.append(score)

        if i_episode % 100 == 0:
            print('Episode {},\r Average score : {} '.format(i_episode,np.mean(scores_window)))


if __name__ =="__main__":

    env = gym.make('SpaceInvaders-v0')
    random_test(env)

    # env = atari_wappers.make_env("SpaceInvaders-v0")
    # state_size, action_size = env.observation_space.shape, env.action_space.n
    # dqn_agent = Agent_dqn(state_size, action_size)
    #
    # dqn_agent.qnetwork_local.load_state_dict(torch.load("dqn_model.pth"))
    # trained_agent_test(env,dqn_agent)



