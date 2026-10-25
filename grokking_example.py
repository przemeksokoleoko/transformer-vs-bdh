import torch, torch.nn as nn, torch.nn.functional as F

p, frac_train = 97, 0.4
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- data ---
a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
a, b = a.flatten(), b.flatten()
eq = torch.full_like(a, p)                       # '=' token id
X = torch.stack([a, b, eq], dim=1)
Y = (a + b) % p
perm = torch.randperm(len(X))
n_tr = int(frac_train * len(X))
tr, te = perm[:n_tr], perm[n_tr:]
Xtr, Ytr, Xte, Yte = [t.to(device) for t in (X[tr], Y[tr], X[te], Y[te])]

# --- model ---
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.attn = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        h = s.ln1(x)
        x = x + s.attn(h, h, h, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))

class Tiny(nn.Module):
    def __init__(s, p, d=128, h=4, L=1):
        super().__init__()
        s.tok = nn.Embedding(p + 1, d)
        s.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        s.blocks = nn.Sequential(*[Block(d, h) for _ in range(L)])
        s.ln = nn.LayerNorm(d)
        s.out = nn.Linear(d, p)
    def forward(s, x):
        x = s.tok(x) + s.pos
        x = s.blocks(x)
        return s.out(s.ln(x)[:, -1])             # predict from '=' position

model = Tiny(p).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

def evaluate(X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return F.cross_entropy(logits, Y).item(), (logits.argmax(-1) == Y).float().mean().item()

for step in range(30000):
    model.train()
    loss = F.cross_entropy(model(Xtr), Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        tl, ta = evaluate(Xtr, Ytr); vl, va = evaluate(Xte, Yte)
        print(f"{step:6d} train {ta:.3f} ({tl:.3f})  test {va:.3f} ({vl:.3f})")
