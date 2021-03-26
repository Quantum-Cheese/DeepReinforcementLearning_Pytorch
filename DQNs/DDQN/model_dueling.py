import torch
import torch.nn as nn
import torch.nn.functional as F

H_1=64
H_2=64

class QNetwork(nn.Module):
    """Dueling Architecture"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.action_size=action_size
        self.seed = torch.manual_seed(seed)
        self.fc1=nn.Linear(state_size,H_1)

        self.fc2_adv = nn.Linear(H_1,H_2)
        self.fc2_v = nn.Linear(H_1, H_2)

        self.fc3_adv = nn.Linear(H_2,action_size)
        self.fc3_v = nn.Linear(H_2, 1)


    def forward(self, state):
        # first hidden layer
        h1=F.relu(self.fc1(state))

        # dueling start in second layer
        h2_adv = F.relu(self.fc2_adv(h1))
        h2_v = F.relu(self.fc2_v(h1))

        # final advantage value
        adv = self.fc3_adv(h2_adv)
        # final state value
        v = self.fc3_v(h2_v).expand(state.size(0), self.action_size) # 从1维扩展到 action_size维

        # calculate final Q(s,a) value for output
        out_q=v+adv-adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)

        return out_q


