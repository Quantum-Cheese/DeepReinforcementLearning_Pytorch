import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from CartPole.Hill_Climbing.agent_HC import hill_climbing


class Policy():
    """
    策略函数是一个单层线性神经网络 P(A)=softmax(W*S)
    输出层加入了激活函数softmax，为了把输出值转换成概率（0-1），但没有中间隐藏层，即没有非线性变换
    输入节点数：s_size ；输出节点数：a_size
    参数矩阵 w 的维度 tate_space x action_space
    """
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action


if __name__=="__main__":
    env = gym.make('CartPole-v0')
    policy=Policy()

    print(policy.w)

    # 训练智能体：更新 policy （参数w）
    scores = hill_climbing(env,policy)

    # 观察训练好的智能体
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for t in range(200):
        action = policy.act(state)
        img.set_data(env.render(mode='rgb_array'))

        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

    # 画累计奖励曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()