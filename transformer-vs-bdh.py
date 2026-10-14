"""
Transformer vs. BDH-GPU ("Dragon Hatchling", arXiv:2509.26507) on a toy task, CPU only.

Task: multi-query associative recall (MQAR).
    k1 v1 k2 v2 ... kN vN   q1 a1 q2 a2 ...
  The first half is a random key->value table that exists ONLY in the context.
  Later, a key is shown again and the model must output its value.
Both models: ~106k parameters, same data, same optimizer, same number of steps.
"""
import argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

N_KEYS, N_VALS = 16, 16
VOCAB = N_KEYS + N_VALS            # keys are 0..15, values are 16..31


def make_batch(B, n_pairs, gen):
    keys = torch.rand(B, N_KEYS, generator=gen).argsort(1)[:, :n_pairs]
    vals = torch.randint(0, N_VALS, (B, n_pairs), generator=gen) + N_KEYS
    pairs = torch.stack([keys, vals], -1).reshape(B, 2 * n_pairs)
    order = torch.rand(B, n_pairs, generator=gen).argsort(1)
    qk, qv = keys.gather(1, order), vals.gather(1, order)
    queries = torch.stack([qk, qv], -1).reshape(B, 2 * n_pairs)
    seq = torch.cat([pairs, queries], 1)
    mask = torch.zeros(B, seq.size(1), dtype=torch.bool)
    mask[:, 2 * n_pairs + 1::2] = True
    return seq, mask[:, 1:]


def ln(x):
    return F.layer_norm(x, (x.size(-1),))


def rope(x, pos, theta=10000.0):
    dh = x.size(-1)
    inv = theta ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, store):
        B, T, d = x.shape
        q, k, v = self.qkv(ln(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(T)
        q, k = rope(q, pos), rope(k, pos)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        att = att.softmax(-1)
        store["attn"].append(att.detach())
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, d))
        hid = F.relu(self.fc1(ln(x)))
        store["mlp_hidden"].append(hid.detach())
        return x + self.fc2(hid)


class Transformer(nn.Module):
    def __init__(self, d=64, h=4, L=2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(L))
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.d, self.L = d, L

    def forward(self, idx):
        self.store = {"attn": [], "mlp_hidden": [], "resid": []}
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.store)
            self.store["resid"].append(x.detach())
        return self.head(ln(x))

    def state_floats(self, T):
        return 2 * self.L * T * self.d


class BDH(nn.Module):
    """
    x = ReLU(v E)                       # D -> N neurons, positive/sparse
    a = LinearAttention(Q=x, K=x, V=v)  # no softmax, queries == keys
    y = ReLU(LN(a) E_v) * x             # gating by already-active neurons
    v = LN(v + LN(y Dec))               # back to the dense stream
    The same weights are reused for all L passes.
    """
    def __init__(self, D=64, H=4, Nh=128, L=2):
        super().__init__()
        self.D, self.H, self.Nh, self.L = D, H, Nh, L
        self.embed = nn.Embedding(VOCAB, D)
        self.enc = nn.Parameter(torch.randn(H, D, Nh) * 0.02)
        self.enc_v = nn.Parameter(torch.randn(H, D, Nh) * 0.02)
        self.dec = nn.Parameter(torch.randn(H * Nh, D) * 0.02)
        self.head = nn.Parameter(torch.randn(D, VOCAB) * 0.02)

    @property
    def n_neurons(self):
        return self.H * self.Nh

    def forward(self, idx):
        B, T = idx.shape
        self.store = {"attn": [], "x": [], "y": []}
        pos = torch.arange(T)
        v = ln(self.embed(idx)).unsqueeze(1)
        for _ in range(self.L):
            x = F.relu(v @ self.enc)
            xr = rope(x, pos)
            scores = (xr @ xr.transpose(-1, -2)).tril(-1)
            a = scores @ v
            y = F.relu(ln(a) @ self.enc_v) * x
            self.store["attn"].append(scores.detach())
            self.store["x"].append(x.detach())
            self.store["y"].append(y.detach())
            y = y.transpose(1, 2).reshape(B, T, self.n_neurons)
            v = ln(v + ln(y @ self.dec).unsqueeze(1))
        return v.squeeze(1) @ self.head

    @torch.no_grad()
    def forward_recurrent(self, idx):
        """Same function, token by token, with a constant-size state per layer."""
        B, T = idx.shape
        state = [torch.zeros(B, self.H, self.Nh, self.D) for _ in range(self.L)]
        out = []
        for t in range(T):
            v = ln(self.embed(idx[:, t])).view(B, 1, 1, self.D)
            for l in range(self.L):
                x = F.relu(v @ self.enc)
                xr = rope(x, torch.tensor([t]))
                a = xr @ state[l]                                   # READ
                state[l] = state[l] + xr.transpose(-1, -2) @ v      # WRITE (Hebbian outer product)
                y = F.relu(ln(a) @ self.enc_v) * x
                y = y.transpose(1, 2).reshape(B, 1, self.n_neurons)
                v = ln(v + ln(y @ self.dec).unsqueeze(1))
            out.append(v.view(B, self.D) @ self.head)
        return torch.stack(out, 1)

    def state_floats(self, T):
        return self.L * self.H * self.Nh * self.D


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def accuracy(model, n_pairs, batches=20, B=64, seed=123):
    gen = torch.Generator().manual_seed(seed)
    ok = tot = 0
    for _ in range(batches):
        seq, mask = make_batch(B, n_pairs, gen)
        pred = model(seq[:, :-1]).argmax(-1)
        ok += (pred[mask] == seq[:, 1:][mask]).sum().item()
        tot += mask.sum().item()
    return ok / tot


def train(model, name, steps, lr, train_pairs=(2, 6), B=32, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    hist, t0 = [], time.time()
    for step in range(steps):
        n = int(torch.randint(train_pairs[0], train_pairs[1] + 1, (1,), generator=gen))
        seq, mask = make_batch(B, n, gen)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits[mask], seq[:, 1:][mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append((step, loss.item()))
        if step % 250 == 0 or step == steps - 1:
            print(f"  [{name:11s}] step {step:5d}  loss {loss.item():.3f}  "
                  f"acc@6 pairs {accuracy(model, 6, batches=4):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    return hist


def frac_active(t):
    return (t > 0).float().mean().item()


@torch.no_grad()
def retrieval_mass(model, n_pairs=8, B=64, seed=7):
    """Per (layer, head): share of |attention| at each query-key position that lands on the
    position of the correct value token."""
    gen = torch.Generator().manual_seed(seed)
    seq, _ = make_batch(B, n_pairs, gen)
    model(seq[:, :-1])
    keys = seq[:, 0:2 * n_pairs:2]
    qpos = torch.arange(2 * n_pairs, 4 * n_pairs, 2)
    qkeys = seq[:, qpos]
    match = (keys.unsqueeze(1) == qkeys.unsqueeze(2)).float().argmax(-1)
    vpos = 2 * match + 1
    res = []
    for att in model.store["attn"]:
        a = att[:, :, qpos, :].abs()
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
        idx = vpos.unsqueeze(1).expand(-1, a.size(1), -1).unsqueeze(-1)
        res.append(a.gather(-1, idx).squeeze(-1).mean((0, 2)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bdh_layers", type=int, default=2)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); tf = Transformer(d=64, h=4, L=2)
    torch.manual_seed(args.seed); bdh = BDH(D=64, H=4, Nh=128, L=args.bdh_layers)

    print("=" * 74); print("MODELS")
    print(f"  Transformer : {n_params(tf):7,d} params | 2 distinct blocks | width 64 | dense, signed")
    print(f"  BDH-GPU     : {n_params(bdh):7,d} params | {args.bdh_layers} passes, SAME weights | "
          f"{bdh.n_neurons} neurons | positive activations")
    print("\nIN-CONTEXT MEMORY SIZE (floats)")
    print(f"  {'context T':>10} | {'Transformer KV cache':>22} | {'BDH state':>12}")
    for T in (32, 128, 512, 2048, 8192):
        print(f"  {T:>10} | {tf.state_floats(T):>22,d} | {bdh.state_floats(T):>12,d}")

    print("\n" + "=" * 74)
    print(f"TRAINING  ({args.steps} steps, batch 32, 2-6 key-value pairs per sequence)")
    h_tf = train(tf, "Transformer", args.steps, args.lr, seed=args.seed)
    h_bdh = train(bdh, "BDH-GPU", args.steps, args.lr, seed=args.seed)

    print("\n" + "=" * 74)
    print("ACCURACY vs. NUMBER OF KEY-VALUE PAIRS   (training saw 2-6; chance = 6%)")
    print(f"  {'pairs':>6} {'seq len':>8} | {'Transformer':>12} | {'BDH-GPU':>8}")
    for n in (2, 4, 6, 8, 10, 12, 16):
        tag = "" if n <= 6 else "  <- longer than training"
        print(f"  {n:>6} {4 * n:>8} | {accuracy(tf, n):>12.3f} | {accuracy(bdh, n):>8.3f}{tag}")

    print("\n" + "=" * 74)
    print("ACTIVATION SPARSITY (fraction of entries > 0), 8 pairs, averaged over layers")
    gen = torch.Generator().manual_seed(5)
    seq, _ = make_batch(64, 8, gen)
    tf(seq[:, :-1]); bdh(seq[:, :-1])
    mlp = sum(frac_active(h) for h in tf.store["mlp_hidden"]) / len(tf.store["mlp_hidden"])
    res = sum(frac_active(r) for r in tf.store["resid"]) / len(tf.store["resid"])
    xs = sum(frac_active(x) for x in bdh.store["x"]) / bdh.L
    ys = sum(frac_active(y) for y in bdh.store["y"]) / bdh.L
    print(f"  Transformer residual stream (signed, dense)  : {res:6.1%} nonzero")
    print(f"  Transformer MLP hidden (ReLU)                : {mlp:6.1%} active")
    print(f"  BDH x  = ReLU(v E)    (attention keys)       : {xs:6.1%} active")
    print(f"  BDH y  = ReLU(.)*x    (output of the layer)  : {ys:6.1%} active")

    print("\n" + "=" * 74)
    print("ATTENTION MASS ON THE CORRECT VALUE TOKEN (best head per layer)")
    for name, m in (("Transformer", tf), ("BDH-GPU", bdh)):
        r = retrieval_mass(m)
        print(f"  {name:11s}: " + "  ".join(f"layer{i + 1}: {x.max():.2f}" for i, x in enumerate(r)))

    print("\n" + "=" * 74)
    print("BDH: PARALLEL FORM vs. RECURRENT FORM (constant-size state)")
    with torch.no_grad():
        a = bdh(seq[:8, :-1]); b = bdh.forward_recurrent(seq[:8, :-1])
    print(f"  max |logit difference| = {(a - b).abs().max().item():.2e}   "
          f"(recurrent state = {bdh.state_floats(0):,d} floats at any T)")

    if args.plot:
        plots(tf, bdh, h_tf, h_bdh)


def plots(tf, bdh, h_tf, h_bdh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(*zip(*h_tf), label="Transformer"); ax.plot(*zip(*h_bdh), label="BDH-GPU")
    ax.set_xlabel("step"); ax.set_ylabel("loss on recalled values"); ax.legend(); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig("training_curves.png", dpi=130)

    gen = torch.Generator().manual_seed(11)
    seq, _ = make_batch(1, 6, gen)
    tf(seq[:, :-1]); bdh(seq[:, :-1])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (name, m) in zip(axs, (("Transformer", tf), ("BDH-GPU", bdh))):
        r = retrieval_mass(m)[-1]; h = int(r.argmax())
        att = m.store["attn"][-1][0, h].abs()
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        ax.imshow(att, cmap="magma")
        ax.set_title(f"{name}: last layer, head {h}\n(row = reading position, col = position read)")
        ax.axhline(11.5, color="c", lw=.6); ax.axvline(11.5, color="c", lw=.6)
        ax.set_xlabel("pairs (left of cyan) | queries")
    fig.tight_layout(); fig.savefig("attention_maps.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(bdh.store["x"][-1].flatten().numpy(), bins=60, alpha=.6, label="BDH x (keys)", log=True)
    ax.hist(tf.store["resid"][-1].flatten().numpy(), bins=60, alpha=.6, label="Transformer residual", log=True)
    ax.legend(); ax.set_xlabel("activation value"); fig.tight_layout(); fig.savefig("activations.png", dpi=130)
    print("saved training_curves.png, attention_maps.png, activations.png")


if __name__ == "__main__":
    main()