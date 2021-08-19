import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from dqn_model import CNN_Model,MLP_Model
from PER_memory import Memory

# from DQNs.DQN_PER.dqn_model import CNN_Model,MLP_Model
# from DQNs.DQN_PER.PER_memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_dqn():
    def __init__(self, input_size,action_size,network_type,eps_decay,decay_type,
                 epsilon = 0.05,tau = 0.001,lr=5e-4,gamma=0.99,
                 update_every = 1,buffer_size=5000,batch_size=32,
                 eps_start=1.0,eps_end=0.01,eps_step = 1000 ,eps_rate =0.997):
        self.input_size = input_size
        self.action_size = action_size

        # Initialize hyper parameters
        self.t_step = 0
        self.episode = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        # setting epsilon
        # 如果未设置衰减策略，则 epsilon 保持固定值
        self.eps = epsilon
        self.eps_decay =eps_decay
        if eps_decay:
            self.eps = eps_start
            self.decay_type = decay_type
            self.eps_end = eps_end
            self.eps_rate = eps_rate
            self.eps_step = eps_step

        # Q network
        if network_type == 'CNN':
            self.qnetwork_local = CNN_Model(input_size, action_size).to(device)
            self.qnetwork_target = CNN_Model(input_size, action_size).to(device)
        else:
            self.qnetwork_local = MLP_Model(input_size, action_size).to(device)
            self.qnetwork_target = MLP_Model(input_size, action_size).to(device)

        # self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr,momentum)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr)

        # Replay memory
        self.per_memory = Memory(buffer_size)

    def epsilon_decay(self):
        if self.decay_type =='linear':
            # epsilon decay
            eps_decrease = (self.eps - self.eps_end) / self.eps_step
            self.eps -= eps_decrease
        elif self.decay_type == 'nonlinear':
            self.eps = max(self.eps_end, self.eps_rate * self.eps)


    def act(self, state, i_episode):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        " Epsilon-greedy action selection"
        if i_episode > self.episode:
            self.episode = i_episode
            if self.eps_decay:
                self.epsilon_decay()
        # epsilon greedy policy
        if random.random() > self.eps:
            action = np.argmax(q_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action,self.eps

    def add_sample(self,sarsd):
        """add every sample and it's corresponding td error to memory"""
        # data transfer, get q values from network
        (state, action, reward, next_state, done) = sarsd

        # reshape state : 3d -> 4d
        dim = list(state.shape)
        dim.insert(0,1)
        state,next_state = np.reshape(state,dim),np.reshape(next_state,dim)
        state_t = torch.from_numpy(state).float().to(device)
        next_state_t = torch.from_numpy(next_state).float().to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            q_estimated = self.qnetwork_local(state_t).data[0][action]

            q_values_next = self.qnetwork_target(next_state_t).data
        self.qnetwork_local.train()

        # add sample to memory
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * torch.max(q_values_next)

        td_error = abs(q_estimated - q_target).item()
        self.per_memory.add(td_error, (state, action, reward, next_state, done))

    def step(self,sarsd):
        self.t_step += 1

        # add an experience for current time step
        self.add_sample(sarsd)

        # train the agent
        if self.per_memory.tree.n_entries > self.batch_size:
            # sample a batch from memory
            mini_batch, idxs, is_weights = self.per_memory.batch_sample(self.batch_size)
            exp_batches = np.array(mini_batch).transpose()

            # Learn every UPDATE_EVERY time steps
            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                if exp_batches.shape[0]!= 5:
                    # print("\n Exception occur during batch sampling !!")
                    return
                self.learn(exp_batches, idxs, is_weights)

    def learn(self,exp_batches,idxs,is_weight):
        # try :
        #     states = torch.tensor(np.vstack(exp_batches[0])).float().to(device)
        # except Exception as e:
        #     print(e)
        #     print(exp_batches.shape)

        # print(type(is_weight))
        # print(is_weight)
        # transfer to tensors
        states = torch.tensor(np.vstack(exp_batches[0])).float().to(device)
        actions = torch.tensor(np.vstack(list(exp_batches[1]))).long().to(device)
        rewards = torch.tensor(np.vstack(list(exp_batches[2]))).float().to(device)
        next_states = torch.tensor(np.vstack(exp_batches[3])).float().to(device)
        dones = torch.tensor(np.vstack(list(exp_batches[4])).astype(np.uint8)).float().to(device)
        is_weight = torch.tensor(is_weight).float().to(device)

        # calculate loss
        Q_expected_next_max = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze( 1)  # shape:[BATCH_SIZE,1](.unsqueeze(1)转换成列向量)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_expected_next_max)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # back prob
        loss = (is_weight * F.mse_loss(Q_targets,Q_expected)).mean()
        self.optimizer.zero_grad()  # 先把原来的梯度清零
        loss.backward()
        self.optimizer.step()

        # get td error
        td_errors = torch.abs(Q_targets - Q_expected).data.cpu().numpy()
        # update priority for per memory
        for i in range(self.batch_size):
            idx = idxs[i]
            self.per_memory.update(idx, td_errors[i])

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


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











