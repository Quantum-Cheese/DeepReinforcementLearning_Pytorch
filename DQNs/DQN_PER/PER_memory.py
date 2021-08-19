import random
import numpy as np
from SumTree import SumTree
# from DQNs.DQN_PER.SumTree import SumTree


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.0001
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    # 根据 TD-error 计算优先级
    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    # 存储一条经验和相应优先级
    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def batch_sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        # beta 随着sample的次数增加而增大（？？），上限为 1.0
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # 把叶子节点分成n个采样区间（n为样本数量）
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # 采样概率
        sampling_probabilities = np.array(priorities) / self.tree.total() + self.e
        # 样本权重： IS weight
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)




