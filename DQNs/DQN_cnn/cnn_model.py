import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import atari_wappers


class CNN_Model (nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN_Model, self).__init__()
        self.conv = nn.Sequential(
            # input_shape 的第一个维度为 输入的 channel 数，比如输入为（4，84，84）时，channel = 4
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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


if __name__ == "__main__":
    env = atari_wappers.make_env("SpaceInvaders-v0")
    state_size, action_size = env.observation_space.shape, env.action_space.n
    print(state_size, action_size)
    model = CNN_Model(state_size, action_size)

    state = env.reset()
    obs = env.reset()
    obs1 = env.reset()
    t = torch.tensor([obs, obs1])
    print("x.shape", t.shape)

    q_value = model.forward(t)
    actions = torch.tensor([[0,1]])
    print(q_value)
    print(q_value.gather(1,actions))
