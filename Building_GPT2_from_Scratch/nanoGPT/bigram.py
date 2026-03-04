import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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
    # 从数据中随机选择batch_size个起始位置，每个位置后面有block_size个字符
    # 之所以是len(data) - block_size，是因为需要保证从起始位置开始的block_size个字符都在数据范围内
    # 例如，如果数据长度是100，block_size是8，那么起始位置的最大值应该是92，因为从位置92开始的8个字符是数据的最后8个字符。
    # randint接收一个范围和一个形状参数，返回在该范围内随机生成的整数张量，形状要求是元组
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    # 从每个起始位置开始，取block_size个字符作为输入x，取从起始位置的下一个字符开始的block_size个字符作为目标y
    # y是指向右偏移一个字符的输入x，表示要预测的下一个字符
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 转化为一个(vocab_size, vocab_size)的矩阵，行是输入字符，列是输出字符，值是对应的logit
        # 我们要预测下一个字符的概率分布，而这个分布的维度就是vocab_size，所以每个输入字符都对应一个长度为vocab_size的向量，表示对每个可能输出字符的logit值。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx和targets都是(B, T)的张量，B是批量大小，T是序列长度
        logits = self.token_embedding_table(idx)  # (B, T, C)，C是vocab_size

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
            logits, loss = self(idx)  # (B, T, C)
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