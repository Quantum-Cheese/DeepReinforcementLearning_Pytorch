"""
对经典 DQN 的改进
1. Double DQN
2. 优先经验回放
"""
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from LunarLander.DQN.model import QNetwork

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-3  # learning rate
UPDATE_EVERY = 4  # how often to update the network
E=1e-8 # small number to add to the priority of experience
ALPHA=0.6


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentV2():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def calculate_prio(self,state, action, reward, next_state, done):
        """calculate priority for given (s,a,r,s')"""
        # state,action,reward,next_state,done=torch.from_numpy(state).float().to(device)\
        #     ,torch.from_numpy(np.array([[action]])).long().to(device)\
        #     ,torch.from_numpy(np.array([reward])).float().to(device)\
        #     ,torch.from_numpy(next_state).float().to(device)\
        #     ,torch.from_numpy(np.array([done]).astype(np.uint8)).float().to(device)
        # Q_targets_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
        # Q_target = reward + (GAMMA * Q_targets_next * (1 - done))
        # Q_expected = self.qnetwork_local(state).gather(1, action).detach()

        state,next_state=torch.from_numpy(state).float().to(device),torch.from_numpy(next_state).float().to(device)
        action=torch.LongTensor([action])

        with torch.no_grad():
            Q_targets_next=self.qnetwork_target(next_state).numpy().max()
            Q_target = reward + (GAMMA * Q_targets_next * (1 - done))

        with torch.no_grad():
            Q_expected = self.qnetwork_local(state).gather(0, action).detach()

        td_error=abs(float(Q_target-Q_expected))
        priority=td_error+E

        return priority

    def step(self, state, action, reward, next_state, done,beta):
        # calculate prb for this experience
        priority=self.calculate_prio(state, action, reward, next_state, done)
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done,priority)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Epsilon 贪婪策略选择动作
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done,weights) tuples
            gamma (float): discount factor
        """
        # 从 experiences 取得所有时间步的 (s,a,r,s',done)的序列，均为列向量 [BATCH_SIZE,1]
        states, actions, rewards, next_states, dones, weights = experiences

        # ------计算每个经验元组对应的Q目标序列
        # 从local网络的 Q estimated 取最大值对应的动作序列
        Q_expected_next_max = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        # shape:[BATCH_SIZE,1](.unsqueeze(1)转换成列向量)

        # 这些动作序列输入target网络得到对应的 Q 估计值，而不是直接让 target 网络选取最大Q（避免了 overestimated 问题）
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_expected_next_max)
        # 根据公式计算 Q 目标
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # --------- Get expected Q values from local model
        # 找到每个 (state,action) 对应的q值，输出为一个q(s,a)序列
        Q_expected = self.qnetwork_local(states).gather(1, actions)  # shape:[BATCH_SIZE,1]

        # --Compute loss
        # modified loss function (multiple with importance sampling weights)
        loss = F.mse_loss(weights*Q_expected, weights*Q_targets) # 用Q估计值和Q目标计算均方差损失函数，都为列向量
        # print("loss:",loss)

        # Minimize the loss
        self.optimizer.zero_grad() # 先把原来的梯度清零
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        # 使用 deque(maxlen=N) 构造函数会新建一个固定大小的队列。当新的元素加入并且这个队列已满的时候， 最老的元素会自动被移除掉
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","priority"])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done,priority):
        """
        Add a new experience to the memory
        :param state:
        :param p: sample probability for this experience
        :return:
        """
        # memory中存入每个经验(s,a,r,s')的采样概率
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self,beta):
        """Randomly sample a batch of experiences from memory."""

        # 先计算 experiences 中每个sample的取样概率
        priorities=[exp.priority for exp in self.memory]
        sum_p=sum([pow(prior,ALPHA) for prior in priorities ])
        probs=[pow(prior,ALPHA)/sum_p for prior in priorities]
        # print("probs:",probs)

        # 计算每个sample的importance-sampling weight
        weights=np.array([pow(1/(self.__len__()*prob),beta) for prob in probs])

        # 根据概率采样
        sample_inds = np.random.choice(self.__len__(), self.batch_size, p=probs, replace=False)
        experiences=[self.memory[ind] for ind in sample_inds]
        # print("sample exps:",experiences)
        # experiences = random.sample(self.memory, k=self.batch_size) # 随机均匀采样

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        weights=torch.from_numpy(weights).float().to(device)

        return (states, actions, rewards, next_states, dones,weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)