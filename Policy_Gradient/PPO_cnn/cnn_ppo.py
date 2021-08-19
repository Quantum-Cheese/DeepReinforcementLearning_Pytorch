import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, output_shape)

