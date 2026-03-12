from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------

'''
c_attn 层，即线性变换层，作用是对输入张量 x 进行线性变换，生成查询（query）、键（key）和值（value）的组合张量。
线性变换的权重矩阵是共享的
'''

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.embd_dim = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is the number of head, hs is the head size, and C = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embd_dim, dim=2)  # (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        # attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # attention output
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y = self.c_proj(y)  # (B, T, C)
        return y

'''
gelu为了解决relu的死神经元问题，在负的位置总会有一些小的梯度
像LLAMA3使用的是SWGELU
'''

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



'''
attention操作相互交流（communicate），而MLP没有交流的过程（单独思考），是映射操作
原transformer的残差路径内部包含归一化，这不太好。我们更倾向于拥有一条从监督开始一直到底层输入的干净路径（clean path），因此我们把layer norm放在残差路径的外面。
这里是预归一化版本，先经过层归一化
'''

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

'''
仿照GPT-2在transformer里的架构
wte：token embedding table
wpe：position embedding table
h：transformer block的列表
ln_f：最后的layer norm
'''

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            wte: nn.Embedding(config.vocab_size, config.n_embd),
            wpe: nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)