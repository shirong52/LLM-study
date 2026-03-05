import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open('/root/LLM-study/Building_GPT2_from_Scratch/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 文本中字符去重，并排序，计算词汇表大小
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 创建字符到索引的映射（stoi）和索引到字符的映射（itos），以及编码和解码函数
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 将文本编码为整数索引的张量，并划分为训练集和验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 数据加载
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()  # 这个装饰器表示在这个函数中不需要计算梯度，节省内存和计算资源
def estimate_loss():
    out = {}
    model.eval()  # 将模型设置为评估模式，禁用dropout等训练特定的行为
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 将模型设置回训练模式
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 注册缓冲区的主要目的是让这个张量在模型保存和加载时可以随着模型的状态一起保存和加载。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # 计算注意力权重
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # 将上三角部分的权重设置为负无穷，确保模型只能关注当前和之前的字符
        wei = F.softmax(wei, dim=-1)  # (B, T, T)，对最后一个维度进行softmax，得到注意力权重

        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 将多个头的输出在最后一个维度(通道维度)上拼接起来 
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),   # 基于每一个token进行的计算
            nn.ReLU(),                    # 激活函数，增加非线性
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(num_heads=n_head, head_size=head_size)  # communication
        self.ffwd = FeedForward(n_embd)  # computation

    def forward(self, x):
        x = self.sa_head(x)  # (B, T, n_embd)
        x = self.ffwd(x)     # (B, T, n_embd)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.Block = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        # self.sa_head = MultiHeadAttention(num_heads=4, head_size=n_embd//4)  # 4个头，每个头的维度是n_embd//4，这样拼接起来的维度就是n_embd
        # self.ffwd = FeedForward(n_embd)  # 前馈神经网络，输入和输出的维度都是n_embd
        self.lm_head = nn.Linear(n_embd, vocab_size) # 经过了线性层才能获得logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx和targets都是(B, T)的张量，B是批量大小，T是序列长度
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        position_emb = self.position_embedding_table(torch.arange(T, device))  # (T, C)
        x = tok_emb + position_emb  # (B, T, C)
        x = self.sa_head(x)  # (B, T, C)
        x = self.ffwd(x)  # (B, T, C)，没有加前馈网络直接进行下一步的话模型没有时间进行思考从其他tokens获得了什么
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # 将logits展平为(B*T, C)，方便计算交叉熵损失，因为交叉熵损失函数期望输入是二维的，其中第一维是样本数量，第二维是类别数量
            targets = targets.view(B*T)  # 将targets展平为(B*T)
            loss = F.cross_entropy(logits, targets)  # 计算交叉熵损失

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx是(B, T)的张量，表示当前上下文的索引数组；max_new_tokens是要生成的新令牌的数量
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # (B, block_size)，取当前上下文的最后block_size个索引，确保输入长度不超过模型的最大上下文长度
            logits, loss = self(idx_cond)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)，取最后一个时间步的logits
            probs = F.softmax(logits, dim=-1)  # (B, C)，计算概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)，从概率分布中采样下一个字符的索引
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)，将新生成的索引添加到当前上下文中
        return idx

model = BigramLanguageModel(vocab_size).to(device)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(max_iters):
    # 获取一个批次的训练数据，xb是输入字符的索引，yb是目标字符的索引
    xb, yb = get_batch('train')

    # logits是模型对输入字符的预测，loss是预测和目标之间的损失
    logits, loss = m(xb, yb)

    # 上一步梯度清零，set_to_none=True可以稍微加速训练；获取参数梯度；更新参数
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 0作为起始token，生成100个新token；
# 索引0取batch的第0个样本；tolist将生成的索引列表转换为Python列表；decode将索引列表转换为字符串
context = torch.zeros((1, 1), dtype=torch.long, device=device) 
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))  