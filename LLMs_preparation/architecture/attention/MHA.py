import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度

        # 初始化qkv投影矩阵，q = xWq, k = xWk, v = xWv
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    '''
    hidden_state形状：(batch_size, sequence_length, hidden_dim)
    例如：[2, 10, 512] 表示 2 条样本，每条 10 个token，每个token 512维向量
    hidden_state 是输入的隐藏状态张量
    '''
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]

        query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_dim)
        key = self.k_linear(hidden_state)    # (batch_size, seq_len, hidden_dim)
        value = self.v_linear(hidden_state)  # (batch_size, seq_len,

        query = self.split_head(query) # (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key)
        value = self.split_head(value)

        # 计算注意力分数
        # (batch_size, num_heads, seq_length, head_dim) -> (batch_size, num_heads, head_dim, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim)) 

        if attention_mask is not None:
            attention_scores += attention_mask * -1e9  # 将mask位置的分数设置为一个非常小的值

        attention_probs = torch.softmax(attention_scores, dim=-1)  # 在最后一个维度上进行softmax

        output = torch.matmul(attention_probs, value)

        output = self.o_linear(output)

        return output

    
    '''
    -1表示自动推断维度大小，num_heads表示分成多少个头，head_dim表示每个头的维度
    transpose把 num_heads 放到第1维，这样后续可以用广播机制并行计算12个头的注意力
    x: (batch_size, seq_len, hidden_size) -> (batch_size, num_heads, seq_len, head_dim)
    '''
    
    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)