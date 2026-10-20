# Transformer vs BDH-GPU

This diagram summarizes the architecture in `transformer-vs-bdh.py`.

```mermaid
flowchart TD
    A["Input tokens"] --> B["Embedding"]
    B --> C["LayerNorm"]

    subgraph T["Transformer block"]
        C --> T1["Q = Wq x"]
        C --> T2["K = Wk x"]
        C --> T3["V = Wv x"]

        T1 -->|Q| T4["RoPE(Q)"]
        T2 -->|K| T4
        T3 -->|V| T5["Masked causal attention"]
        T4 --> T5
        T5 --> T6["Attn output"]
        T6 --> T7["Proj + residual"]
        T7 --> T8["LayerNorm"]
        T8 --> T9["MLP: FC -> ReLU -> FC"]
        T9 --> T10["Residual + output"]
        T10 --> T11["Head"]
    end

    T11 --> T12["Logits"]

    subgraph B["BDH-GPU layer"]
        C2["LayerNorm"] --> D1["v = LN(embed)"]
        D1 --> D2["x = ReLU(v @ enc)"]
        D2 --> D3["xr = RoPE(x)"]
        D3 --> D4["scores = xr @ xr^T (causal)"]
        D1 --> D4
        D4 --> D5["a = scores @ v"]
        D5 --> D6["y = ReLU(LN(a) @ enc_v) * x"]
        D6 --> D7["y -> reshape"]
        D1 --> D8["v = LN(v + LN(y @ dec))"]
        D7 --> D8
        D8 --> D9["Next BDH pass"]
    end

    D9 --> D10["Head"]
    D10 --> D11["Logits"]

    classDef trans fill:#dfeeff,stroke:#1f4e79,stroke-width:1.5px,color:#000;
    classDef bdh fill:#e8f7e8,stroke:#2e7d32,stroke-width:1.5px,color:#000;
    classDef base fill:#f4f4f4,stroke:#444,stroke-width:1px,color:#000;

    class A,B,C,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12 trans;
    class C2,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11 bdh;
    class B base;
```

## Key idea

- Transformer: explicit token-to-token attention with a growing KV cache.
- BDH-GPU: sparse key activations and recurrent-style memory read/write updates.

## Suggested filename

`transformer-vs-bdh-mermaid.md`
