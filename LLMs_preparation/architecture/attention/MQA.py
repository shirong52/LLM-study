# 区别在于：query仍然进行分头，和多头注意力机制相同，而key和value只有一个头
# 标准多头注意力:  K, V 都是 [batch, num_heads, seq_len, head_dim]  → 每个头有独立的 K, V
# 多查询注意力:    K, V 都是 [batch, 1, seq_len, head_dim]          → 所有头共享同一组 K, V

import torch
from torch import nn

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiQueryAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_size, attention_mask=None):
        batch_size = hidden_size.size()[0]

        # 线性变换
        query = self.q_linear(hidden_size)
        key = self.k_linear(hidden_size)
        value = self.v_linear(hidden_size)

        query = self.split_heads(query)
        key = self.split_heads(key, 1)  # key和value只有一个头，(batch, 1, seq_len, head_dim)
        value = self.split_heads(value, 1)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))

        if attention_mask is not None:
            attention_scores += attention_mask * -1e9

        attention_probs = torch.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_probs, value)

        # (batch, num_heads, seq_len, head_dim) → (batch, num_heads, head_dim, seq_len)，确保连续内存
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.hidden_size)

        output = self.o_linear(output)
        return output

    def split_heads(self, x, head_num=None):
        batch_size = x.size()[0]

        if head_num is None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)