import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from cnn_model import CNN_Model

TAU = 1e-3  # for soft update of target parameters
EPS_start=1.0
EPS_end=0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the memory
        :param state:
        :param p: sample probability for this experience
        :return:
        """
        # memory中存入每个经验(s,a,r,s')的采样概率
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor([e.state for e in experiences if e is not None]).float().to(device)
        actions = torch.tensor([[e.action for e in experiences if e is not None]]).long().to(device)
        rewards = torch.tensor([e.reward for e in experiences if e is not None]).float().to(device)
        next_states = torch.tensor([e.next_state for e in experiences if e is not None]).float().to(
            device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent_dqn():
    def __init__(self, input_channel,action_size,learning_rate=5e-3,buffer_size=1e4,batch_size=32):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = CNN_Model(input_channel,action_size).to(device)
        self.qnetwork_target = CNN_Model(input_channel,action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), learning_rate)

        # Replay memory
        self.batch_size = batch_size
        self.memory = ReplayBuffer(action_size, buffer_size,batch_size)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.episode = 0
        self.epsilon = EPS_start

    def act(self,state,i_episode,eps_decay):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        " Epsilon-greedy action selection"
        if i_episode>self.episode:
            # update EPS every new episode
            self.epsilon = max(EPS_end, eps_decay * self.epsilon)
            self.episode = i_episode
        # epsilon greedy policy
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def act_greedy_policy(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def step(self,sarsd,gamma,update_every):
        state, action, reward, next_state, done = sarsd
        self.t_step += 1

        # add an experience for current time step
        self.memory.add(state,action,reward,next_state,done)

        # Learn every UPDATE_EVERY time steps
        if (self.t_step+1) % update_every==0:
            if self.memory.__len__()>self.batch_size:
                batch_exps = self.memory.sample()
                loss = self.learn(batch_exps,gamma)
                return loss

    def learn(self,exps,gamma):
        # fetch the batch (s,a,r,s',done) from experiences batch
        states,actions,rewards,next_states,dones = exps
        # print(states.device)

        # ------------------ calculate loss —------------------------- #

        # calculate Q targets
        expected_next_max_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(0)
        Q_expected_next = self.qnetwork_target(next_states).gather(1, expected_next_max_actions)
        Q_targets = rewards + (gamma * Q_expected_next * (1 - dones))

        # get expected Q for current state
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        # ---------------- update local Q net -------------------- #
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(next(self.qnetwork_local.parameters()).is_cuda)

        # ---------------- update target Q net -------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        return loss.cpu().detach().numpy()


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









