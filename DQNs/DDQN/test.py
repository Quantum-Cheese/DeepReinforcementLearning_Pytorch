import gym
import numpy as np
import torch
from collections import namedtuple, deque


target_org=np.array([[ 0.0910, -0.0224, -0.0552, -0.0192],
        [ 0.0908, -0.0209, -0.0553, -0.0181],
        [ 0.0922, -0.0219, -0.0546, -0.0206],
        [ 0.0913, -0.0211, -0.0548, -0.0182],
        [ 0.0910, -0.0211, -0.0554, -0.0187]])
target_org=torch.tensor(target_org)
# print(target_org.shape)
# # 按行取最大值
# print(target_org.detach().max(1))
# print(target_org.detach().max(1)[0])
# # 转换成列向量
# print(target_org.detach().max(1)[0].unsqueeze(1))


local_org=np.array([[ 0.0936, -0.0768, -0.1730, -0.0238],
        [ 0.0930, -0.0620, -0.1845, -0.0077],
        [ 0.0986, -0.0473, -0.1868,  0.0110],
        [ 0.0946, -0.0752, -0.1726, -0.0264],
        [ 0.0979, -0.0497, -0.1886,  0.0097]])
local_org=torch.tensor(local_org)
actions=torch.tensor(np.array(
    [[3],
    [1],
    [2],
    [0],
    [0]]))
# print(actions)
# print(local_org.shape)
# print(local_org.gather(1, actions.long()))


b=torch.tensor(np.array([ 0.0932, -0.0206, -0.0541, -0.0204]))
action=torch.LongTensor([0])
print(b.gather(0,action))

memory=deque(maxlen=10)
exp=namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
e1=exp(0.34,1,3.56,0.56,False)
memory.append(e1)
e2=exp(3.34,0,8.56,-2.3,False)
memory.append(e2)
memory.append(exp(4.6,0,8.56,-2.3,False))
memory.append(exp(8.7,0,8.56,-4.3,False))
memory.append(exp(2.2,0,-0.8,-2.3,False))

# print(memory)
# print(memory[0].state)
# print(len(memory))
# #
# sample_inds=np.random.choice(len(memory), 3, p=[0.1,0.2,0.2,0.4,0.1],replace=False)
# print(sample_inds)


# env = gym.make('LunarLander-v2')
# env.seed(0)
# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)




