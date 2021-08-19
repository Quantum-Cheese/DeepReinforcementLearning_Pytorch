import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP_Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(MLP_Model, self).__init__()
        self.fc1=nn.Linear(state_size,128)
        self.fc2=nn.Linear(128,256)
        self.fc3=nn.Linear(256,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        out=self.fc1(state)
        out=F.relu(out)
        out=self.fc2(out)
        out=F.relu(out)
        q_a=self.fc3(out)

        return q_a


class CNN_Model (nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN_Model, self).__init__()
        self.conv = nn.Sequential(
            # input_shape 的第一个维度为 输入的 channel 数，比如输入为（4，84，84）时，channel = 4
            nn.Conv2d(input_shape[0], 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, input_shape):
        o = self.conv(torch.zeros((1, *input_shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.fc(conv_out)
