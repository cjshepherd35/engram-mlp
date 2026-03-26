import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('device is: ', device)

#parameters to tweak
max_iters =  7_001  #1_001
eval_iters = 100
eval_interval =  1_000
n_embed = 64   #64
block_size = 64
batch_size = 12
learning_rate = 5e-3
n_head = 4  #4
n_layer = 6  #6
dropout = 0.2 

vocab_size = 500 #my own preset number, may need changing
num_merges = vocab_size - 256 #256 is how many distinct utf-8 tokens there are.

tokens = text.encode("utf-8")
tokens = list(map(int, tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i<len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+= 2
        else:
            newids.append(ids[i])
            i+=1
    return newids

ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("merged")
print('len: ',len(ids))

vocab = {idx: bytes([idx]) for  idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors='replace')
    return text

def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break #nothing else can be merged.
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for  i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

# --- Engram Implementation ---

engram_layer_ids = [2,6] # Applying Engram module only to layer index 1
engram_max_ngram_size = 3
engram_vocab_size = [1024, 1024]
engram_n_embed_per_ngram = n_embed
engram_n_head_per_ngram = 4
engram_kernel_size = 4

class EngramMLP(nn.Module):
    def __init__(self, vocab_size, n_embed, max_ngram_size, engram_n_embed_per_ngram):
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.n_embed = n_embed
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.embedding = nn.Embedding(vocab_size, n_embed)
        
        self.mlps = nn.ModuleList()
        for n in range(2, max_ngram_size + 1):
            mlp = nn.Sequential(
                nn.Linear(n * n_embed, 4 * n_embed),
                nn.SiLU(),
                nn.Linear(4 * n_embed, engram_n_embed_per_ngram)
            )
            self.mlps.append(mlp)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        
        results = []
        for i, n in enumerate(range(2, self.max_ngram_size + 1)):
            pad = torch.zeros(B, n-1, self.n_embed, device=x.device)
            x_padded = torch.cat([pad, x], dim=1)
            
            # Using unfold to get sliding windows of size n
            windows = x_padded.unfold(1, n, 1) # (B, T, n_embed, n)
            windows = windows.transpose(2, 3) # (B, T, n, n_embed)
            windows = windows.reshape(B, T, n * self.n_embed)
            
            res = self.mlps[i](windows)
            results.append(res)
            
        return torch.cat(results, dim=-1)


class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size=4, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).contiguous()
        return y

class EngramLayer(nn.Module):
    def __init__(self, layer_id, n_embed):
        super().__init__()
        self.layer_id = layer_id
        
        self.engram_mlp = EngramMLP(
            vocab_size=vocab_size,
            n_embed=n_embed,
            max_ngram_size=engram_max_ngram_size,
            engram_n_embed_per_ngram=engram_n_embed_per_ngram
        )
        self.short_conv = ShortConv(
            hidden_size=n_embed,
            kernel_size=engram_kernel_size,
            dilation=engram_max_ngram_size,
        )
        engram_hidden_size = (engram_max_ngram_size - 1) * engram_n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, n_embed)
        self.key_proj = nn.Linear(engram_hidden_size, n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        
    def forward(self, hidden_states, input_ids):
        embeddings = self.engram_mlp(input_ids)

        
        key = self.key_proj(embeddings)
        normed_key = self.norm1(key)
        normed_query = self.norm2(hidden_states)
        
        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(n_embed)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate).unsqueeze(-1)
        
        value = gate * self.value_proj(embeddings)
        output = value + self.short_conv(value)
        return output

# --- End Engram Implementation ---

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(), 
            nn.Linear(4*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head, layer_id):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.engram = EngramLayer(layer_id, n_embed) if layer_id in engram_layer_ids else None
        
    def forward(self, x, idx):
        if self.engram is not None:
            x = x + self.engram(x, idx)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed, n_head=n_head, layer_id=i) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) 

    def forward(self, idx, targets=None):
        b,t = idx.shape
        token_embed = self.token_embedding_table(idx) #(b,t,c)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device)) #also (b,t,c)
        x = pos_embed + token_embed
        for block in self.blocks:
            x = block(x, idx)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b,t,c = logits.shape
            logits = logits.view(b*t,c)
            targets = targets.view(b*t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
   
if __name__ == '__main__':
    model = Transformer()
    total_params = sum(p.numel() for p in model.parameters())
    print('size of model',total_params)
    m = model.to(device)
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        if not iter % eval_interval:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # torch.save(m.state_dict(), 'engramkarp_model.pth')
    # print("Model saved to engramkarp_model.pth")
    
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))