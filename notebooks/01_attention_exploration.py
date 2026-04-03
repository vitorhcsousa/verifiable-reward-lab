import marimo

__generated_with = "0.22.0"
app = marimo.App(
    width="medium",
    layout_file="layouts/01_attention_exploration.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Attention Mechanism — The Complete Visual Guide

    An interactive walkthrough of scaled dot-product attention and multi-head
    attention. Every slider change re-executes the full pipeline — watch
    shapes, scores, and attention patterns respond in real time.

    > **How to use:** drag the sliders in the Configuration section below,
    > then scroll down. Every visualization updates automatically.

    **Code:** [`src/rlvr_from_scratch/model/attention.py`](https://github.com/vitorhcsousa/rlvr-from-scratch)
    **Article:** [Attention Is All You Need to Implement](https://www.vitorsousa.com/foundations/attention-from-scratch)
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    import lovely_tensors as lt
    lt.monkey_patch()
    return F, go, make_subplots, math, nn, np, torch


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    sl_d = mo.ui.slider(start=16, stop=128, step=16, value=64, label="d_model", full_width=True)
    sl_h = mo.ui.slider(start=1, stop=8, step=1, value=4, label="n_heads", full_width=True)
    sl_t = mo.ui.slider(start=4, stop=16, step=1, value=8, label="Sequence length T", full_width=True)
    sl_b = mo.ui.slider(start=1, stop=4, step=1, value=2, label="Batch size B", full_width=True)
    sw_causal = mo.ui.switch(value=True, label="Causal mask")
    sl_seed = mo.ui.slider(start=0, stop=99, step=1, value=42, label="Seed", full_width=True)

    mo.hstack(
        [
            mo.vstack([sl_d, sl_h, sl_t], align="stretch"),
            mo.vstack([sl_b, sw_causal, sl_seed], align="stretch"),
        ],
        justify="start",
        gap=2,
    )
    return sl_b, sl_d, sl_h, sl_seed, sl_t, sw_causal


@app.cell
def _(math, mo, sl_b, sl_d, sl_h, sl_seed, sl_t, sw_causal):
    B = sl_b.value
    T = sl_t.value
    d_model = sl_d.value
    n_heads = sl_h.value
    d_k = d_model // n_heads
    use_causal = sw_causal.value
    seed = sl_seed.value

    mo.hstack(
        [
            mo.stat(value=B, label="Batch", bordered=True),
            mo.stat(value=T, label="Seq len", bordered=True),
            mo.stat(value=d_model, label="d_model", bordered=True),
            mo.stat(value=n_heads, label="Heads", bordered=True),
            mo.stat(value=d_k, label="d_k", bordered=True),
            mo.stat(value=f"{1/math.sqrt(d_k):.3f}", label="Scale (1/√d_k)", bordered=True),
            mo.stat(value=f"{4*d_model**2:,}", label="Params", bordered=True),
        ],
        justify="center",
        gap=0.5,
    )
    return B, T, d_k, d_model, n_heads, seed, use_causal


@app.cell
def _(mo):
    mo.md("""
    ---
    ## The Pipeline at a Glance
    """)
    return


@app.cell
def _(B, T, d_k, d_model, mo, n_heads):
    mo.mermaid(
        f"""
        graph LR
            X["Input<br/>({B}, {T}, {d_model})"]
            X --> WQ["W_Q"]
            X --> WK["W_K"]
            X --> WV["W_V"]
            WQ --> Split["Split Heads<br/>({B}, {n_heads}, {T}, {d_k})"]
            WK --> Split
            WV --> Split
            Split --> Score["QK^T / √{d_k}<br/>({B}, {n_heads}, {T}, {T})"]
            Score --> Mask["+ Mask"]
            Mask --> SM["Softmax"]
            SM --> Agg["× V"]
            Agg --> Merge["Merge Heads<br/>({B}, {T}, {d_model})"]
            Merge --> WO["W_O"]
            WO --> Out["Output<br/>({B}, {T}, {d_model})"]

            style X fill:#1e293b,stroke:#06b6d4,color:#cffafe
            style Score fill:#1e293b,stroke:#f59e0b,color:#fde68a
            style SM fill:#1e293b,stroke:#8b5cf6,color:#e9d5ff
            style Out fill:#1e293b,stroke:#10b981,color:#d1fae5
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 1 — Input & Projections

    Every token gets three learned representations:

    $$Q = XW^Q \qquad K = XW^K \qquad V = XW^V$$

    - **Q** (query): what am I looking for?
    - **K** (key): what do I advertise?
    - **V** (value): what do I return if selected?
    """)
    return


@app.cell
def _(B, T, d_model, nn, seed, torch):
    torch.manual_seed(seed)
    X = torch.randn(B, T, d_model)
    _W_Q = nn.Linear(d_model, d_model, bias=False)
    _W_K = nn.Linear(d_model, d_model, bias=False)
    _W_V = nn.Linear(d_model, d_model, bias=False)
    Q = _W_Q(X)
    K = _W_K(X)
    V = _W_V(X)
    print(f"X = {X!r}")
    print(f"Q = {Q!r}")
    print(f"K = {K!r}")
    print(f"V = {V!r}")
    return K, Q, V, X


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Q, K, V are three *learned views* of the same input.** The model learns what to search for, what to advertise, and what to return — independently."),
        kind="info",
    )
    return


@app.cell
def _(d_k, d_model, mo, n_heads):
    mo.md(f"""
    ---
    ## Step 2 — Split Into {n_heads} Heads

    Reshape `(B, T, {d_model})` → `(B, {n_heads}, T, {d_k})`. **Zero cost — just a view.**
    Each head sees a different {d_k}-dim slice.
    """)
    return


@app.cell
def _(K, Q, V, n_heads):
    def _split(x, h):
        _B, _T, _ = x.shape
        _dk = x.shape[-1] // h
        return x.view(_B, _T, h, _dk).transpose(1, 2)
    Q_h = _split(Q, n_heads)
    K_h = _split(K, n_heads)
    V_h = _split(V, n_heads)
    print(f"Q_h = {Q_h!r}")
    return K_h, Q_h, V_h


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Step 3 — Score: $QK^T$

    Dot product between every query and every key. Entry $(i, j)$ = how much
    token $i$ should attend to token $j$.
    """)
    return


@app.cell
def _(K_h, Q_h, torch):
    scores_raw = torch.matmul(Q_h, K_h.transpose(-2, -1))
    print(f"scores = {scores_raw!r}")
    return (scores_raw,)


@app.cell
def _(go, mo, n_heads, scores_raw):
    _batch_idx = 0
    _tabs = {}
    for h in range(n_heads):
        _data = scores_raw[_batch_idx, h].detach().cpu().numpy()
        _fig = go.Figure(
            data=go.Heatmap(
                z=_data[::-1],
                colorscale="Magma",
                text=[[f"{v:.2f}" for v in row] for row in _data[::-1]],
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate="Query %{y} → Key %{x}<br>Score: %{z:.3f}<extra></extra>",
            )
        )
        _T = _data.shape[0]
        _fig.update_layout(
            title=f"Head {h} — Raw Scores (before scaling)",
            xaxis_title="Key position",
            yaxis_title="Query position",
            yaxis=dict(tickvals=list(range(_T)), ticktext=list(range(_T - 1, -1, -1))),
            template="plotly_dark",
            height=400,
            margin=dict(t=40, b=40),
        )
        _tabs[f"Head {h}"] = mo.ui.plotly(_fig)

    mo.tabs(_tabs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Step 4 — Why Scale by $\sqrt{d_k}$?

    This is not optional. Without scaling, attention breaks.
    """)
    return


@app.cell
def _(d_k, math, mo):
    mo.accordion(
        {
            "Full derivation (click to expand)": mo.md(
                rf"""
    For random $q, k \in \mathbb{{R}}^{{d_k}}$ with entries from $\mathcal{{N}}(0, 1)$:

    $$q \cdot k = \sum_{{i=1}}^{{d_k}} q_i k_i$$

    Each $q_i k_i$ has mean $0$ and variance $1$ (product of independent standard normals).

    Since the ${d_k}$ terms are independent:

    $$\text{{Var}}(q \cdot k) = d_k = {d_k}$$

    Standard deviation = $\sqrt{{{d_k}}}$ = **{math.sqrt(d_k):.2f}**

    Softmax on inputs with std ≈ {math.sqrt(d_k):.1f} produces near-one-hot outputs.
    Gradients through saturated softmax ≈ 0. Training stalls.

    Dividing by $\sqrt{{d_k}}$ normalizes variance to $1$:

    $$\text{{Var}}\left(\frac{{q \cdot k}}{{\sqrt{{d_k}}}}\right) = \frac{{d_k}}{{d_k}} = 1$$
                """
            ),
        }
    )
    return


@app.cell
def _(F, d_k, go, make_subplots, math, mo, scores_raw, torch):
    _n = 50_000
    _q = torch.randn(_n, d_k)
    _k = torch.randn(_n, d_k)
    _dots = (_q * _k).sum(-1)
    _dots_s = _dots / math.sqrt(d_k)

    # Softmax comparison
    _w_raw = F.softmax(scores_raw[0, 0].detach(), dim=-1)
    _scores_s = scores_raw / math.sqrt(d_k)
    _w_scl = F.softmax(_scores_s[0, 0].detach(), dim=-1)

    _ent_raw = -(_w_raw * _w_raw.clamp(min=1e-9).log()).sum(-1).mean().item()
    _ent_scl = -(_w_scl * _w_scl.clamp(min=1e-9).log()).sum(-1).mean().item()
    _max_ent = math.log(scores_raw.shape[-1])

    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Unscaled · Var={_dots.var():.1f}",
            f"Scaled (÷√{d_k}) · Var={_dots_s.var():.2f}",
            "Attention Entropy",
        ],
    )
    fig4.add_trace(go.Histogram(x=_dots.numpy(), nbinsx=80, marker_color="#06b6d4", opacity=0.85, name="Unscaled"), row=1, col=1)
    fig4.add_trace(go.Histogram(x=_dots_s.numpy(), nbinsx=80, marker_color="#10b981", opacity=0.85, name="Scaled"), row=1, col=2)
    fig4.add_trace(
        go.Bar(
            x=["Unscaled", "Scaled", "Uniform (max)"],
            y=[_ent_raw, _ent_scl, _max_ent],
            marker_color=["#ef4444", "#10b981", "#64748b"],
            text=[f"{v:.2f}" for v in [_ent_raw, _ent_scl, _max_ent]],
            textposition="outside",
            name="Entropy",
        ),
        row=1, col=3,
    )
    fig4.update_layout(template="plotly_dark", height=350, showlegend=False, margin=dict(t=50, b=30))
    mo.ui.plotly(fig4)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"**The $\sqrt{d_k}$ scaling is derived from statistics, not tuned.** Without it, softmax saturates, gradients vanish, and the model cannot learn attention patterns."),
        kind="warn",
    )
    return


@app.cell
def _(d_k, math, scores_raw):
    scores_scaled = scores_raw / math.sqrt(d_k)
    print(f"scores_scaled = {scores_scaled!r}")
    return (scores_scaled,)


@app.cell
def _(mo, use_causal):
    _status = "**ON** — token $i$ only sees tokens $\\leq i$" if use_causal else "**OFF** — every token sees everything"
    mo.md(f"---\n## Step 5 — Causal Mask ({_status})\n\nAdditive convention: `0.0` = allowed, `-inf` = blocked. After softmax, $e^{{-\\infty}} = 0$.")
    return


@app.cell
def _(T, go, mo, np, scores_scaled, torch, use_causal):
    if use_causal:
        _m = torch.triu(torch.ones(T, T), diagonal=1).bool()
        mask = _m.float().masked_fill(_m, float("-inf")).unsqueeze(0).unsqueeze(0)
    else:
        mask = torch.zeros(1, 1, T, T)

    scores_masked = scores_scaled + mask

    # Visualize mask
    _mask_display = mask[0, 0].clone().numpy()
    _text = np.where(np.isinf(_mask_display), "-∞", "0")

    _fig_mask = go.Figure(
        data=go.Heatmap(
            z=np.where(np.isinf(_mask_display), -1, 0)[::-1],
            colorscale=[[0, "#1e293b"], [0.5, "#1e293b"], [1, "#ef4444"]],
            text=_text[::-1],
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            showscale=False,
            hovertemplate="Query %{y} → Key %{x}<br>%{text}<extra></extra>",
        )
    )
    _fig_mask.update_layout(
        title="Causal Mask" if use_causal else "No Mask",
        xaxis_title="Key position", yaxis_title="Query position",
        yaxis=dict(tickvals=list(range(T)), ticktext=list(range(T - 1, -1, -1))),
        template="plotly_dark", height=380, width=420, margin=dict(t=40, b=40),
    )
    mo.ui.plotly(_fig_mask)
    return mask, scores_masked


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Step 6 — Softmax → Attention Weights

    Each row becomes a probability distribution: non-negative, sums to 1.
    **Hover** over cells to see exact values. Switch between heads with tabs.
    """)
    return


@app.cell
def _(F, scores_masked):
    weights = F.softmax(scores_masked, dim=-1)
    print(f"weights = {weights!r}")
    _sums = weights.sum(dim=-1)
    print(f"Row sums: min={_sums.min():.6f}  max={_sums.max():.6f}")
    return (weights,)


@app.cell
def _(go, make_subplots, mo, n_heads, weights):
    def _():
        _tabs_w = {}
        for h in range(n_heads):
            _w = weights[0, h].detach().cpu().numpy()
            _T = _w.shape[0]
            _fig = go.Figure(
                data=go.Heatmap(
                    z=_w[::-1],
                    colorscale="Magma",
                    zmin=0,
                    zmax=_w.max(),
                    text=[[f"{v:.2f}" for v in row] for row in _w[::-1]],
                    texttemplate="%{text}",
                    textfont={"size": 9},
                    hovertemplate="Query %{y} → Key %{x}<br>Weight: %{z:.4f}<extra></extra>",
                )
            )
            _fig.update_layout(
                title=f"Head {h} — Attention Weights",
                xaxis_title="Key position",
                yaxis_title="Query position",
                yaxis=dict(tickvals=list(range(_T)), ticktext=list(range(_T - 1, -1, -1))),
                template="plotly_dark",
                height=420,
                margin=dict(t=40, b=40),
            )
            _tabs_w[f"Head {h}"] = mo.ui.plotly(_fig)

        # Add "All Heads" comparison
        _fig_all = make_subplots(rows=1, cols=n_heads, subplot_titles=[f"H{h}" for h in range(n_heads)])
        for h in range(n_heads):
            _w = weights[0, h].detach().cpu().numpy()
            _fig_all.add_trace(
                go.Heatmap(z=_w[::-1], colorscale="Magma", zmin=0, zmax=_w.max(), showscale=False),
                row=1, col=h + 1,
            )
        _fig_all.update_layout(template="plotly_dark", height=350, margin=dict(t=40, b=30))
        _tabs_w["All Heads"] = mo.ui.plotly(_fig_all)
        return mo.tabs(_tabs_w)


    _()
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md("**Each head learns a different attention pattern** — some focus locally, some attend broadly. The model discovers this specialization without explicit supervision."),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 7 — Token Attention Explorer

    Pick a query token and see what it attends to across all heads.
    """)
    return


@app.cell
def _(T, mo):
    sl_query_token = mo.ui.slider(start=0, stop=T - 1, step=1, value=T - 1, label="Query token", full_width=True)
    sl_query_token
    return (sl_query_token,)


@app.cell
def _(go, make_subplots, mo, n_heads, sl_query_token, weights):
    def _():
        _qi = sl_query_token.value
        _fig_tok = make_subplots(
            rows=1, cols=1,
            subplot_titles=[f"Token {_qi} attends to..."],
        )
        _colors = ["#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#64748b", "#84cc16"]
        for h in range(n_heads):
            _row = weights[0, h, _qi].detach().cpu().numpy()
            _fig_tok.add_trace(
                go.Bar(
                    x=list(range(len(_row))),
                    y=_row,
                    name=f"Head {h}",
                    marker_color=_colors[h % len(_colors)],
                    opacity=0.85,
                )
            )
        _fig_tok.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Key position",
            yaxis_title="Attention weight",
            barmode="group",
            margin=dict(t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        return mo.ui.plotly(_fig_tok)


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Step 8 — Weighted Sum & Merge Heads

    $$\text{output}_i = \sum_j w_{ij} \cdot v_j$$

    Then transpose, `.contiguous()`, reshape back to $(B, T, d_\text{model})$, apply $W^O$.
    """)
    return


@app.cell
def _(V_h, d_model, nn, seed, torch, weights):
    attn_out = torch.matmul(weights, V_h)
    print(f"attn_out = {attn_out!r}")

    _B, _H, _T, _dk = attn_out.shape
    merged = attn_out.transpose(1, 2).contiguous().view(_B, _T, _H * _dk)
    print(f"merged   = {merged!r}")

    torch.manual_seed(seed + 99)
    _W_O = nn.Linear(d_model, d_model, bias=False)
    final_output = _W_O(merged)
    print(f"output   = {final_output!r}")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 9 — Full Module

    The complete `MultiHeadAttention` from the repo, inspected with `torchinfo`.
    """)
    return


@app.cell
def _(X, d_model, mask, n_heads, seed, torch):
    from rlvr_from_scratch.model.attention import MultiHeadAttention
    from torchinfo import summary as ti_summary

    torch.manual_seed(seed)
    mha = MultiHeadAttention(d_model, n_heads)
    mha_out, mha_w, _ = mha(X, X, X, mask=mask)
    print(f"output  = {mha_out!r}")
    print(f"weights = {mha_w!r}")
    return mha, ti_summary


@app.cell
def _(X, mha, ti_summary):
    ti_summary(mha, input_data=(X, X, X), depth=2, col_names=["input_size", "output_size", "num_params"], verbose=0)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Step 10 — KV-Cache: Incremental Decoding

    During generation, we decode one token at a time. KV-cache avoids
    recomputing K and V for previous tokens.

    **Invariant:** incremental output must exactly match full forward pass.

    Use the slider to step through generation and watch the cache grow.
    """)
    return


@app.cell
def _(T, mo):
    sl_gen_step = mo.ui.slider(start=1, stop=T, step=1, value=T, label="Generation steps", full_width=True)
    sl_gen_step
    return (sl_gen_step,)


@app.cell
def _(B, T, X, d_k, go, mha, mo, n_heads, sl_gen_step, torch):
    from rlvr_from_scratch.model.attention import causal_mask as _mk_mask

    _steps = sl_gen_step.value
    mha.eval()

    # Full pass (ground truth)
    with torch.no_grad():
        _full_out, _, _ = mha(X, X, X, mask=_mk_mask(T))

    # Incremental pass up to _steps
    _cache = (torch.empty(B, n_heads, 0, d_k), torch.empty(B, n_heads, 0, d_k))
    _inc_parts = []
    _cache_sizes = []
    with torch.no_grad():
        for t in range(_steps):
            _tok = X[:, t : t + 1, :]
            _out, _, _cache = mha(_tok, _tok, _tok, kv_cache=_cache)
            _inc_parts.append(_out)
            _cache_sizes.append(_cache[0].shape[2])

    _inc_out = torch.cat(_inc_parts, dim=1)
    _diff = (_full_out[:, :_steps] - _inc_out).abs().max().item()

    # Cache growth chart
    _fig_cache = go.Figure()
    _fig_cache.add_trace(go.Bar(
        x=list(range(_steps)),
        y=_cache_sizes,
        marker_color="#10b981",
        text=[str(s) for s in _cache_sizes],
        textposition="outside",
        hovertemplate="After token %{x}: cache = %{y} keys<extra></extra>",
    ))
    _fig_cache.add_trace(go.Scatter(
        x=list(range(_steps)),
        y=[d_k * s * n_heads * 2 for s in _cache_sizes],
        mode="lines+markers",
        name="Memory (floats)",
        yaxis="y2",
        line=dict(color="#f59e0b", dash="dot"),
        hovertemplate="Memory: %{y:,} floats<extra></extra>",
    ))
    _fig_cache.update_layout(
        title=f"KV-Cache Growth ({_steps}/{T} tokens decoded)",
        xaxis_title="Decoding step",
        yaxis_title="Cached keys",
        yaxis2=dict(title="Memory (floats)", overlaying="y", side="right", showgrid=False),
        template="plotly_dark",
        height=350,
        margin=dict(t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    mo.vstack([
        mo.hstack([
            mo.stat(value=_steps, label="Tokens decoded", bordered=True),
            mo.stat(value=_cache[0].shape[2], label="Keys cached", bordered=True),
            mo.stat(value=f"{_diff:.1e}", label="Max diff vs full", bordered=True),
            mo.stat(
                value="✓" if _diff < 1e-4 else "✗",
                label="Match",
                bordered=True,
            ),
        ], justify="center", gap=0.5),
        mo.ui.plotly(_fig_cache),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Step 11 — Gradient Flow

    All paths must carry gradients — inputs and all four projections.
    """)
    return


@app.cell
def _(B, T, d_model, mha, mo, torch):
    from rlvr_from_scratch.model.attention import causal_mask as _mk_mask3

    mha.train()
    _X_g = torch.randn(B, T, d_model, requires_grad=True)
    _o, _, _ = mha(_X_g, _X_g, _X_g, mask=_mk_mask3(T))
    _o.sum().backward()

    _rows = []
    _rows.append(("X (input)", _X_g.grad is not None, f"{_X_g.grad.norm():.4f}"))
    for n, p in mha.named_parameters():
        _ok = p.grad is not None and p.grad.abs().sum() > 0
        _rows.append((f"`{n}`", _ok, f"{p.grad.norm():.4f}"))

    _table = "| Parameter | Gradient | Norm |\n|---|---|---|\n"
    for name, ok, norm in _rows:
        _table += f"| {name} | {'✓' if ok else '✗'} | {norm} |\n"

    mo.md(_table)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | Operation | What to watch |
    |------|-----------|---------------|
    | 1 | Project Q, K, V | Three views of same input |
    | 2 | Split heads | Zero-cost reshape, each head gets $d_k$ dims |
    | 3 | $QK^T$ | $T \times T$ score matrix per head |
    | 4 | Scale $\div \sqrt{d_k}$ | Prevents softmax saturation |
    | 5 | Causal mask | Upper triangle → $-\infty$ |
    | 6 | Softmax | Rows sum to 1, future = 0 |
    | 7 | $\text{weights} \times V$ | Weighted retrieval from values |
    | 8 | Merge + $W^O$ | Back to $(B, T, d_\text{model})$ |

    **Three matrix multiplies and a softmax.** Everything else is engineering.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
