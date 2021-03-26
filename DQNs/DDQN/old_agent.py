import numpy as np
import random
from collections import namedtuple, deque
from torch.autograd import Variable
from Pro_LunarLander.model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
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
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()

                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # 根据当前状态，获取值函数 q(s,a)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():  # 用于不需要梯度的情景，在这里用训练过的网络进行推断，输出给定state的q values
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()  # 切换为训练模式

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # -------取得经验元组中的所有经验训练Q网络-------#
        states, actions, rewards, next_states, dones = experiences

        # --- 获取用于损失函数的output序列
        # 获取 local 网络输出的 q values 序列
        q_values=self.qnetwork_local.forward(states) # 包括每个state上，所有action的q value
        # print("Q values shape:",q_values.shape)

        # 对经验元组中的每一对（s,a）,找到对应的q(s,a)值，构建新的 q值序列，只包括当前state下实际采取action的q值
        qa_values=[]
        for index,action in enumerate(actions):
            ac=action.numpy()[0]
            qa_value=q_values[index,ac].detach().numpy()  # 把tensor转换为numpy（这里直接拿到了float)
            qa_values.append(qa_value)
        outputs_trans=torch.from_numpy(np.array(qa_values))
        # print("output shape:",outputs_trans.shape)

        # --- 计算Q目标序列，作为均方误差损失函数中的 targets
        qhat_maxes=[]
        # 对每一个 next_state 放入目标网络输出相应的q value,取最大的value
        for next_state,done in zip(next_states,dones):
            if not done:
                with torch.no_grad():  # 用于不需要梯度的情景，在这里用训练过的网络进行推断，输出给定state的q values
                    action_values = self.qnetwork_target.forward(next_state)
                qhat_maxes.append(action_values.numpy().max())
            else:
                qhat_maxes.append(0)

        qhat_maxes=torch.from_numpy(np.array(qhat_maxes)) # 转换为tensor

        targets=rewards.flatten()+gamma*qhat_maxes  # 根据公式计算 Q 目标
        # print("targets shape:",targets.shape)

        # ------------------------训练------------------------------------#
        # 创建 Variable
        outputTransV=Variable(outputs_trans,requires_grad=True)
        targetsV=Variable(targets,requires_grad=True)

        self.optimizer.zero_grad()
        # 定义损失函数
        loss = F.mse_loss(outputTransV, targetsV) # 均方误差
        # 反向传播更新参数
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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


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
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)