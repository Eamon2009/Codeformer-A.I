import torch
import torch.nn as nn
from torch.nn import functional as F
import time
from config.config import *

start = time.time()

# Hyperparameters
batch_size    = 64
block_size    = 128
max_iters     = 10000
eval_interval = 100
learning_rate = 1e-3
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters    = 200
n_embd        = 64
n_head        = 4
n_layer       = 4
dropout       = 0.0
# ------------

with open(cleaned_path, 'r', encoding='utf-8') as f:
    text2 = f.read()

chars2     = sorted(list(set(text2)))
vocab_size = len(chars2)
stri       = {ch: i for i, ch in enumerate(chars2)}
it         = {i: ch for i, ch in enumerate(chars2)}
encode     = lambda s: [stri[c] for c in s]       # string -> list of ints
decode     = lambda l: ''.join([it[i] for i in l]) # list of ints -> string

data = torch.tensor(encode(text2), dtype=torch.long)

# Split into train and validation sets
n          = int(train_split * len(data))
train_data = data[:n]
val_data   = data[n:]

torch.manual_seed(seed)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,))
    x    = torch.stack([data[i:i + block_size]         for i in ix])
    y    = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split)
            _, loss    = model(X, Y)
            losses[k]  = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5          # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)                        # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        v   = self.value(x)  # (B, T, head_size)
        out = wei @ v        # (B, T, head_size)
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _    = self(idx)
            logits       = logits[:, -1, :]                          # (B, C)
            probs        = F.softmax(logits, dim=-1)                 # (B, C)
            idx_next     = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx          = torch.cat((idx, idx_next), dim=1)         # (B, T+1)
        return idx


# Initialise model
model = BigramLanguageModel(vocab_size).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

end = time.time()
print("Time taken:", end - start)