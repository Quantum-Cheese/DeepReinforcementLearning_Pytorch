import math
import torch.multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from Actor_Critic.A3C.untils import ValueNetwork,ActorDiscrete,ActorContinous


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Worker(mp.Process):
    def __init__(self,env,continuous,state_size,action_size,id, global_valueNet,global_value_optimizer,
                 global_policyNet,global_policy_optimizer,
                 global_epi,global_epi_rew,rew_queue,
                 max_epi,gamma):
        super(Worker, self).__init__()
        # define env for individual worker
        self.env = env
        self.continuous = continuous
        self.name = str(id)
        self.env.seed(id)
        self.state_size = state_size
        self.action_size = action_size
        self.memory=[]

        # passing global settings to worker
        self.global_valueNet,self.global_value_optimizer = global_valueNet,global_value_optimizer
        self.global_policyNet,self.global_policy_optimizer = global_policyNet,global_policy_optimizer
        self.global_epi,self.global_epi_rew = global_epi,global_epi_rew
        self.rew_queue = rew_queue
        self.max_epi = max_epi
        # self.batch_size = batch_size
        self.gamma = gamma

        # define local net for individual worker
        self.local_policyNet = ActorDiscrete(self.state_size,self.action_size).to(device)
        if self.continuous:
            self.local_policyNet = ActorContinous(self.state_size,self.action_size).to(device)
        self.local_valueNet = ValueNetwork(self.state_size,1).to(device)

    def sync_global(self):
        self.local_valueNet.load_state_dict(self.global_valueNet.state_dict())
        self.local_policyNet.load_state_dict(self.global_policyNet.state_dict())

    def calculate_loss(self):
        # get experiences from current trajectory
        states = torch.tensor([t[0] for t in self.memory], dtype=torch.float)
        log_probs = torch.tensor([t[1] for t in self.memory], dtype=torch.float)

        # -- calculate discount future rewards for every time step
        rewards = [t[2] for t in self.memory]
        fur_Rewards = []
        for i in range(len(rewards)):
            discount = [self.gamma ** i for i in range(len(rewards) - i)]
            f_rewards = rewards[i:]
            fur_Rewards.append(sum(d * f for d, f in zip(discount, f_rewards)))
        fur_Rewards = torch.tensor(fur_Rewards, dtype=torch.float).view(-1, 1)

        # calculate loss for critic
        V = self.local_valueNet(states)
        value_loss = F.mse_loss(fur_Rewards, V)

        # compute entropy for policy loss
        (mu, sigma) = self.local_policyNet(states)
        dist = Normal(mu, sigma)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(dist.scale)  # exploration

        # calculate loss for actor
        advantage = (fur_Rewards - V).detach()
        policy_loss = -advantage * log_probs
        policy_loss = (policy_loss - 0.005 * entropy).mean()

        return value_loss,policy_loss

    def update_global(self):
        value_loss, policy_loss = self.calculate_loss()

        self.global_value_optimizer.zero_grad()
        value_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_valueNet.parameters(), self.global_valueNet.parameters()):
            global_params._grad = local_params._grad
        self.global_value_optimizer.step()

        self.global_policy_optimizer.zero_grad()
        policy_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_policyNet.parameters(), self.global_policyNet.parameters()):
            global_params._grad = local_params._grad
        self.global_policy_optimizer.step()

        self.memory=[]  # clear trajectory

    def run(self):
        while self.global_epi.value < self.max_epi:
            state = self.env.reset()
            total_reward=0
            while True:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                action, prob = self.local_policyNet.act(state)  # 离散空间取直接prob，连续空间取log prob
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append([state,action,reward,next_state,done])
                total_reward += reward
                state = next_state

                if done:
                    # recoding global episode and episode reward
                    with self.global_epi.get_lock():
                        self.global_epi.value += 1
                    with self.global_epi_rew.get_lock():
                        if self.global_epi_rew.value == 0.:
                            self.global_epi_rew.value = total_reward
                        else:
                            # Moving average reward
                            self.global_epi_rew.value = self.global_epi_rew.value * 0.99 + total_reward * 0.01
                    self.rew_queue.put(self.global_epi_rew.value)

                    print("w{} | episode: {}\t , episode reward:{:.4} \t  "
                          .format(self.name,self.global_epi.value,self.global_epi_rew.value))
                    break

            # update and sync with the global net when finishing an episode
            self.update_global()
            self.sync_global()

        self.rew_queue.put(None)



