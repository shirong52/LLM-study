import torch
from labml_nn.sampling import Sampler

class TopKSampler(Sampler):
    def __init__(self, k: int, sampler: Sampler):
        self.k = k
        self.sampler = sampler

    
    # __call__ 是 Python 的特殊方法（魔法方法），它让一个类的实例可以像函数一样被调用。
    def __call__(self, logits: torch.Tensor):
        # 创建与 logits 同形状的全1张量，并将其乘以负无穷大
        zeros = logits.new_zeros(logits.shape) * float('-inf')
        # 在vocab_size维度上进行操作，从 logits 张量中找出最大的 k 个值及其对应的索引位置
        values, indices = torch.topk(logits, self.k, dim=-1)
        # 将 logits 中最大的 k 个值设置为 value，其他值保持负无穷大
        zeros.scatter_(-1, indices, values)

        return self.sampler(zeros)