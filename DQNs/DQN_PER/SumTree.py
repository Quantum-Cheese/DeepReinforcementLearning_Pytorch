import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    # 从叶子节点到根节点向上传播，更新整棵树
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # 更新目标节点的 priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # 存储样本和对应节点的 priority （只有叶子节点可以存储，上面节点的值都是下层的求和）
    def add(self, p, data):
        # 计算叶子节点的 index
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        # 如果叶子节点已满，则从第一个开始清空重新存储
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # 从根节点开始搜索，找到对应的叶子节点
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # 采样方法，取得样本和对应的 priority
    def get(self, s):
        # 找到叶子节点的索引
        idx = self._retrieve(0, s) # s：在每个区间随机取的值
        # 找到样本的索引
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
