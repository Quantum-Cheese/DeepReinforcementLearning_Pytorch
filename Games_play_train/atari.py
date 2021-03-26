import gym
import os
from skimage import io

env = gym.make('SpaceInvaders-v0')
#env = gym.make("PongDeterministic-v4")
status = env.reset()


print('observation space:', env.observation_space)
print('action space:', env.action_space)


def save_films(state,step):
    if not os.path.exists('./image'):
        os.makedirs('./image')
    img_name = './image/pic-%d.jpg' % step
    io.imsave(img_name, state)


for step in range(5000):
    env.render()
    action =1
    state, reward, done, info = env.step(action)

    if step%100 ==0 :
        print(state.shape)
        # print(state)
        save_films(state,step)

    if reward >0:
        print(reward,step)
        save_films(state,step)

    if done:
        print('dead in %d steps' % step)
        break