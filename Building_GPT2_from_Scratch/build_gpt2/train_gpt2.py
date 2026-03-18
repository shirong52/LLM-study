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
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

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

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"  # T必须小于等于block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)，位置嵌入
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None: 
            # 计算交叉熵损失，logits的形状是(B, T, vocab_size)，targets的形状是(B, T)，我们需要将它们展平为(B*T, vocab_size)和(B*T)的形状来计算损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 
    
    def from_pretrained(cls, model_type):
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']  # GPT-2的四个版本
        from transformers import GPT2LMHeadModel
        print(f"Loading pretrained weights for {model_type}...")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # 预训练权重中有一些参数是GPT-2特有的，我们不需要它们，将其过滤掉
        sd_keys = sd.keys() 
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # GPT-2没有attn.bias参数

        model_df = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_df.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # GPT-2没有attn.masked_bias参数
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # GPT-2没有attn.bias参数

        # 原本用的tensorflow版本的GPT-2权重是转置的，而huggingface版本的权重是正常的，因此我们需要对某些权重进行转置才能正确加载
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']  # 硬编码了需要被转置的权重

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys:{len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(x) for x in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


import tiktoken
enc = tiktoken.get_encoding("gpt2")  # 获取GPT-2的编码器
with open('/root/LLM-study/Building_GPT2_from_Scratch/build_gpt2/input.txt', 'r') as f:
    text = f.read()
text = text[:1000]  # 取前1000个字符进行测试
tokens = enc.encode(text)  # 将文本编码为token ID
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)  # 并非有状态的，不会自动转到device，而是返回指向设备上新内存的指针，要用等号
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

device = 'cpu'

model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item()}") #loss是包含单个元素的张量，item()将元素转换为一个单精度浮点数

print(loss)
import sys;sys.exit(0)

new_return_sequences = 5
max_length = 30

# model = GPT.from_pretrained('gpt2')
# model.eval()  
# model.to('cuda')

import tiktoken
enc = tiktoken.get_encoding("gpt2")  
tokens = enc.encode("Hello, I'm a language model,")  # 将文本编码为token ID
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).report(num_return_sequences, 1)
x = tokens.to('cuda')

torch.manual_seed(42) 
torch.cuda.manual_seed_all(42)
# 一直输出token，直到达到最大长度
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # 取最后一个时间步的logits，形状为(B, vocab_size)
        probs = F.softmax(logits, dim=-1)  # 计算softmax概率
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  
        ix = torch.multinomial(topk_probs, num_samples=1)  # 从topk概率中采样一个token ID，形状为(B, 1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)  # 将采样的索引转换为实际的token ID，形状为(B, 1)
        x = torch.cat((x, xcol), dim=1)  # 连接上当前token，形状为(B, T+1)


for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()  # 取出生成的token ID，形状为(max_length,)
    decoded = enc.decode(tokens)  # 将token ID解码为文本
    prin(">", decoded)