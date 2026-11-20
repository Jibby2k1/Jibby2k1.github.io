---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Multiple heads and masking

:::{admonition} What you will learn
:class: tip
1. How multi-head attention works. The width is split into $H$ heads, each head attends on its own, and the
   results are concatenated. You will check that this is exactly $H$ independent single-head attentions, at the
   same parameter count as one head.
2. Why several heads can do what one head cannot. One softmax row is one probability distribution, so one head
   cannot look in two places at once. On a toy task you will watch two heads split the neighbourhood between
   them.
3. How a padding mask lets neighbour sets of different sizes share one batch. You will prove that padded tokens
   get weight exactly zero, that the output does not depend on the padded values (or their gradients), and
   that it matches the unpadded computation.
4. How an additive float mask generalises the boolean mask, which is where model B puts its geometry, and why
   the causal mask of language models has no place in a model of particles.
:::

**Prerequisites:** [Lesson 05b](05b_attention_from_scratch.md) (queries, keys, values and one head of
scaled dot-product attention). [Lesson 05a](05a_shielding.md) explains why we want attention between
neighbours at all. The transformer chapter of the original paper {cite}`vaswani2017attention` is a good
companion read.

```{code-cell} ipython3
:tags: [remove-cell]
import sys, pathlib
_root = pathlib.Path.cwd().resolve()
while not (_root / "course_utils").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))
import course_utils as cu
cu.set_style()
```

```{code-cell} ipython3
import copy, math, os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import torch
import torch.nn.functional as F

from course_utils import metrics as mt, plotting as cplt
from course_utils.attention import (attention, softmax, split_heads, merge_heads, causal_mask, padding_mask,
                                    MultiHeadSelfAttention)
from course_utils.equivariance import check_permutation, report
from course_utils.models import param_count

torch.set_num_threads(min(4, os.cpu_count() or 1))   # small models: a few threads are enough
T0 = time.time()
TOL = 1e-12
```

## 1. One head is one weighted average

In Lesson 05b, every token $j$ (here: the query point or one of its neighbours) produced a query
$\mathbf Q_j = W_Q\mathbf h_j$, a key $\mathbf K_j = W_K\mathbf h_j$ and a value $\mathbf V_j = W_V\mathbf h_j$,
and attention replaced token $j$ by

$$
\mathbf a_j = \sum_k w_{jk}\,\mathbf V_k, \qquad
w_{jk} = \operatorname{softmax}_k(s_{jk}), \qquad
s_{jk} = \frac{\mathbf Q_j\cdot\mathbf K_k}{\sqrt d}.
$$

As in the rest of Part V, $j$ is the token that looks and $k$ the token it looks at: $s_{jk}$ is a score and
$w_{jk}$ an attention weight.

Read the weights row by row. For a fixed token $j$, the numbers $w_{j1},\dots,w_{jL}$ are positive and add up to
one. They form **one probability distribution** over the tokens, and $\mathbf a_j$ is the average of the values
under that distribution. The value vector has many channels, so one head can carry many numbers at once. But all
of them are averaged with the *same* weights and divided by the *same* total. Suppose the next layer needs "the
mean of $v$ over group A" and "the mean of $v$ over group B", where the two groups have different and varying
sizes. One distribution can be sharp on group A, or on group B, or broad over both; it cannot normalise by the
size of group A in some channels and by the size of group B in others. Each head's softmax does exactly one
such normalisation, for free. Two heads do two.

<!-- TODO(animation): a multi-head version of S5 would fit here: one query particle, two coloured heads,
     each lighting up its own half of the neighbourhood. No such scene exists yet (S1-S6 cover other topics). -->

## 2. Multi-head attention: split, attend, concatenate

The fix is to run several distributions side by side. Take tokens $\mathbf h_j \in \mathbb R^{D}$, where $D$ is
the model width. In code, a set of $L$ tokens is stored as a matrix with one token per **row**,
$\mathbf h \in \mathbb R^{L\times D}$, so the projection $\mathbf Q_j = W_Q\mathbf h_j$ of Lesson 05b becomes
`h @ Wq` with `Wq` $= W_Q^\top$ (same numbers, rows instead of columns). Multi-head attention with $H$ heads does
four things:

1. **Project** as usual: $\mathbf Q = \mathbf h W_Q^\top$, $\mathbf K = \mathbf h W_K^\top$,
   $\mathbf V = \mathbf h W_V^\top$, each $L\times D$.
2. **Split** the $D$ columns into $H$ blocks of $d_h = D/H$ columns. Block $h$ gives $\mathbf Q^{(h)},
   \mathbf K^{(h)}, \mathbf V^{(h)} \in \mathbb R^{L\times d_h}$. The head dimension $d_h$ plays the role of $d$
   in Lesson 05b, including in the scale $1/\sqrt{d_h}$.
3. **Attend** in every head on its own, with its own softmax:
   $\mathbf A^{(h)} = \operatorname{softmax}\big(\mathbf Q^{(h)}\mathbf K^{(h)\top}/\sqrt{d_h}\big)\,\mathbf V^{(h)}$.
4. **Concatenate** the heads back to width $D$ and mix them with one more matrix:
   $\operatorname{MHA}(\mathbf h) = \big[\mathbf A^{(1)}\,|\,\cdots\,|\,\mathbf A^{(H)}\big]\,W_O^\top$.

Nothing new is learned per head. $W_Q$, $W_K$, $W_V$ and $W_O$ are the same four $D\times D$ matrices that a
single head of width $D$ uses. The only change is that the scores are computed blockwise, so there are $H$
softmax rows per token instead of one. In code, the split is a `reshape` and the concatenation is its inverse.

```{code-cell} ipython3
def mha_inline(h, Wq, Wk, Wv, Wo, H, mask=None):
    """Multi-head self-attention written with reshapes. h (B, L, D) -> (B, L, D); Wq etc. are D x D (= W_Q^T)."""
    B, L, W = h.shape
    d = W // H
    split = lambda t: t.reshape(B, L, H, d).transpose(1, 2)          # (B, H, L, d_h)
    q, k, v = split(h @ Wq), split(h @ Wk), split(h @ Wv)
    logits = q @ k.transpose(-2, -1) / math.sqrt(d)                    # (B, H, L, L): one score matrix per head
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    w = torch.softmax(logits, dim=-1)
    a = w @ v                                                          # (B, H, L, d_h)
    return a.transpose(1, 2).reshape(B, L, W) @ Wo, w                  # concatenate, then mix


def mha_loop(h, Wq, Wk, Wv, Wo, H):
    """The same thing as H separate single-head attentions on column blocks, concatenated."""
    W = h.shape[-1]
    d = W // H
    outs = []
    for i in range(H):
        cols = slice(i * d, (i + 1) * d)
        outs.append(attention(h @ Wq[:, cols], h @ Wk[:, cols], h @ Wv[:, cols]))
    return torch.cat(outs, dim=-1) @ Wo


g = torch.Generator().manual_seed(0)
B_, L_, W_, H_ = 3, 7, 8, 2
h = torch.randn(B_, L_, W_, generator=g, dtype=torch.float64)
Wq, Wk, Wv, Wo = (torch.randn(W_, W_, generator=g, dtype=torch.float64) / math.sqrt(W_) for _ in range(4))

out_inline, w_inline = mha_inline(h, Wq, Wk, Wv, Wo, H_)
out_loop = mha_loop(h, Wq, Wk, Wv, Wo, H_)
print(report("reshape version vs loop over heads", float((out_inline - out_loop).abs().max()), TOL))
```

The reshapes must agree with the course helpers `split_heads` and `merge_heads`, and merging must undo
splitting exactly. Both are pure index shuffles, so the comparison is exact, with no round-off at all:

```{code-cell} ipython3
q_ours = (h @ Wq).reshape(B_, L_, H_, W_ // H_).transpose(1, 2)
print("split_heads matches our reshape exactly:", torch.equal(split_heads(h @ Wq, H_), q_ours))
print("merge_heads(split_heads(x)) == x exactly:", torch.equal(merge_heads(split_heads(h, H_)), h))
```

The course's `MultiHeadSelfAttention` and PyTorch's `nn.MultiheadAttention` store the same four matrices in
their own layouts (one fused $D\times 3D$ input projection, plus biases we set to zero here). Copy ours in and all
three must agree to round-off:

```{code-cell} ipython3
cu_mha = MultiHeadSelfAttention(W_, H_).double()
th_mha = torch.nn.MultiheadAttention(W_, H_, bias=False, batch_first=True).double()
with torch.no_grad():
    cu_mha.qkv.weight.copy_(torch.cat([Wq, Wk, Wv], dim=1).T)    # nn.Linear computes x @ weight.T
    cu_mha.qkv.bias.zero_()
    cu_mha.out.weight.copy_(Wo.T)
    cu_mha.out.bias.zero_()
    th_mha.in_proj_weight.copy_(torch.cat([Wq, Wk, Wv], dim=1).T)
    th_mha.out_proj.weight.copy_(Wo.T)
    out_cu = cu_mha(h)
    out_th, _ = th_mha(h, h, h, need_weights=False)
print(report("ours vs course_utils MultiHeadSelfAttention", float((out_inline - out_cu).abs().max()), TOL))
print(report("ours vs torch.nn.MultiheadAttention", float((out_inline - out_th).abs().max()), TOL))
```

The parameter count does not depend on $H$:

```{code-cell} ipython3
for H in (1, 2, 4, 8):
    print(f"width 16, {H} head(s): {param_count(MultiHeadSelfAttention(16, H)):,} parameters")
```

Here is the split drawn out for one set of tokens. The $D = 8$ columns of $\mathbf Q$ are cut into two blocks,
each block gets its own score matrix and its own softmax (middle), and the head outputs are laid side by side
again (right) before $W_O$ mixes them.

```{code-cell} ipython3
:tags: [hide-input]
fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), layout="constrained",
                         gridspec_kw=dict(width_ratios=[1.05, 1, 1, 1.05]))
head_cols = [cplt.PALETTE[0], cplt.PALETTE[1]]
d = W_ // H_
qmat = (h @ Wq)[0].numpy()
a_heads = (w_inline[0] @ split_heads(h @ Wv, H_)[0]).numpy()          # (H, L, d_h)
amat = np.concatenate(list(a_heads), axis=1)
for ax, mat, title, clab in ((axes[0], qmat, r"$\mathbf{Q}$: $L \times D$, split into heads", "entry of Q"),
                             (axes[3], amat, r"concatenated $[\mathbf{A}^{(1)} | \mathbf{A}^{(2)}]$", "entry of A")):
    lim = float(np.abs(mat).max())
    im = ax.imshow(mat, cmap=cplt.CMAP_DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label=clab)
    ax.grid(False)
    for i in range(H_):
        ax.add_patch(Rectangle((i * d - 0.5, -1.25), d, 0.6, color=head_cols[i], clip_on=False))
        ax.text(i * d + d / 2 - 0.5, -1.6, f"head {i + 1}", ha="center", va="bottom", color=head_cols[i])
    ax.axvline(d - 0.5, color=cplt.INK, lw=1.5, ls="--")
    ax.set_ylim(L_ - 0.5, -1.9)
    ax.set_xlabel("channel"), ax.set_ylabel("token")
    ax.set_title(title, fontsize=10, pad=18)
for i in range(H_):
    cplt.attention_heatmap(w_inline[0, i], ax=axes[1 + i], colorbar=False, vmax=1.0,
                           title=f"head {i + 1}: its own softmax", xlabel="key k", ylabel="query j")
    for s in axes[1 + i].spines.values():
        s.set_visible(True), s.set_color(head_cols[i]), s.set_linewidth(2.5)
fig.colorbar(plt.cm.ScalarMappable(cmap=cplt.CMAP_SEQUENTIAL, norm=plt.Normalize(0, 1)), ax=axes[1:3],
             shrink=0.8, label="attention weight")
plt.show()
```

Different heads produce different attention patterns from the same tokens, because each head sees a different
slice of the projected features. Training decides what each slice should encode. The next section gives the
heads a reason to differ.

Multi-head attention still treats the tokens as a set: relabelling the inputs relabels the outputs in the same
way (Lesson [05d](05d_permutation.md) explains why). A quick check:

```{code-cell} ipython3
check_permutation(lambda x: mha_inline(x, Wq, Wk, Wv, Wo, H_)[0], (h,), axis=1, mode="equivariant",
                  name="multi-head self-attention");
```

## 3. A task that one head cannot do

We build a toy task in which the neighbourhood of a query point naturally splits in two. The suspensions come
from `course_utils.toy_data`. The particles settle along a fixed lab direction $\hat{\mathbf g}$ (the toy's
settling direction). For a query point $\mathbf x_f$ we keep every particle whose centre lies within a
**cutoff radius** $r_c$. With $\mathbf r_j$ the centre of neighbour $j$ and $\mathbf x_{fj} = \mathbf x_f -
\mathbf r_j$ the offset from that centre to the query point (as everywhere in the course), we describe neighbour
$j$ by three numbers:

- $c_j = -\hat{\mathbf g}\cdot\mathbf x_{fj}/|\mathbf x_{fj}|$, the cosine of the angle between the settling
  direction and the direction from the point to the neighbour. $c_j > 0$ means the neighbour is **ahead** of the
  point (further along the direction of settling), $c_j < 0$ means **behind**;
- its distance $r_j = |\mathbf x_{fj}|$;
- a value $v_j = |\mathbf q_j|$, the magnitude of the particle's state (its slip speed).

The target has two components: the mean of $v_j$ over the neighbours ahead, and the mean over the neighbours
behind,

$$
y_{\text{ahead}} = \frac{\sum_{j:\,c_j>0} v_j}{\#\{j: c_j>0\}}, \qquad
y_{\text{behind}} = \frac{\sum_{j:\,c_j<0} v_j}{\#\{j: c_j<0\}} .
$$

This is a made-up target, not physics. It isolates the one thing we want to see: an output that needs two
different averages, each normalised by its own, varying group size, over the same set. Two remarks before the code.

- **The counts vary.** With a cutoff, one query point has only a couple of neighbours and the next has a dozen. A batch must be a
  rectangular tensor, so we pad every set to the same length `N_MAX` and record which slots are real. Section 5
  is about doing that correctly. For now, take the mask as given.
- **This task is not rotation-invariant**, on purpose: $c_j$ refers to a fixed lab axis. Lesson
  [05f](05f_invariant_attention.md) builds attention whose inputs are all rotation invariants.

```{code-cell} ipython3
R_CUT, N_MAX = 2.0, 16

t = time.time()
snaps = cu.make_snapshots(5)
print(f"generated 5 snapshots in {time.time() - t:.1f} s; {snaps[0].n} spheres each")
g_hat = torch.tensor(snaps[0].config.settling_direction, dtype=torch.float64)


def neighbour_sets(snap, n_points, seed):
    """Variable-size neighbour sets (within R_CUT), padded to N_MAX, nearest first."""
    pts = snap.sample_exterior(n_points, rng=np.random.default_rng(seed))
    inp = cu.neighbour_inputs(pts, snap.positions, snap.slip, snap.phi_local, snap.density_ratio, N_MAX)
    x = inp["x_fj"]                                          # x_f - r_j, (P, N_MAX, 3), nearest first
    r = x.norm(dim=-1)
    c = (-x / r[..., None]) @ g_hat                          # cos(angle between g and r_j - x_f)
    v = inp["scalars"][..., 0]                               # |q_j|
    real = r < R_CUT                                         # (P, N_MAX) True for real neighbours
    ahead, behind = real & (c > 0), real & (c < 0)
    keep = (ahead.sum(1) > 0) & (behind.sum(1) > 0)          # both averages must exist
    y = torch.stack([(v * ahead).sum(1) / ahead.sum(1).clamp_min(1),
                     (v * behind).sum(1) / behind.sum(1).clamp_min(1)], -1)
    return dict(c=c[keep], r=r[keep], v=v[keep], real=real[keep], y=y[keep], x_fj=x[keep])


def stack(sets):
    return {k: torch.cat([s[k] for s in sets]) for k in sets[0]}


train_raw = stack([neighbour_sets(s, 800, seed=i) for i, s in enumerate(snaps[:4])])
test_raw = stack([neighbour_sets(snaps[4], 800, seed=4)])
counts = train_raw["real"].sum(1)
print(f"train sets {len(counts)}, test sets {len(test_raw['real'])}; "
      f"neighbours per set: min {counts.min().item()}, mean {counts.float().mean():.1f}, max {counts.max().item()}")
assert counts.max() < N_MAX, "every set must fit, with room to spare, so that no neighbour is cut off"
```

The real neighbours are always the first slots, because `neighbour_inputs` sorts them nearest first. So the real
part of each row is a prefix, and the course helper `padding_mask(lengths, n)` builds the same mask from the
counts alone:

```{code-cell} ipython3
same = torch.equal(padding_mask(counts, N_MAX)[:, 0, 0], train_raw["real"])
print("padding_mask(lengths) reproduces the cutoff mask:", same)
```

The features enter the network standardised (zero mean, unit spread over the real training tokens). Padded slots
are filled with zeros, an arbitrary choice that Section 5 will show does not matter.

```{code-cell} ipython3
real_tr = train_raw["real"]
STATS = {k: (train_raw[k][real_tr].mean(), train_raw[k][real_tr].std()) for k in ("r", "v")}


def features(raw):
    """(P, N_MAX, 3) = [c, (r - mean)/std, (v - mean)/std], zeros in padded slots."""
    f = torch.stack([raw["c"]] + [(raw[k] - STATS[k][0]) / STATS[k][1] for k in ("r", "v")], -1)
    return torch.where(raw["real"][..., None], f, torch.zeros_like(f))


train_f, test_f = features(train_raw), features(test_raw)
print("feature tensor:", tuple(train_f.shape), "| mask:", tuple(train_raw["real"].shape))
```

```{code-cell} ipython3
:tags: [hide-input]
def meridional(raw, i):
    """Neighbours of set i in the (along g, across g) half-plane: axial and radial distance from the point."""
    m = raw["real"][i]
    rr, cc = raw["r"][i][m].numpy(), raw["c"][i][m].numpy()
    return rr * cc, rr * np.sqrt(np.clip(1 - cc ** 2, 0, None)), m


def draw_neighbourhood(ax, raw, i, weights=None, title=None, cmap=cplt.CMAP_SEQUENTIAL, vmax=None):
    along, across, m = meridional(raw, i)
    ax.add_patch(Rectangle((0, 0), R_CUT, R_CUT, color=cplt.PALETTE[1], alpha=0.10, lw=0))
    ax.add_patch(Rectangle((-R_CUT, 0), R_CUT, R_CUT, color=cplt.PALETTE[0], alpha=0.10, lw=0))
    th = np.linspace(0, np.pi, 100)
    ax.plot(R_CUT * np.cos(th), R_CUT * np.sin(th), color=cplt.INK, lw=1, ls="--")
    ax.axvline(0, color=cplt.INK, lw=0.8)
    if weights is None:
        cols = [cplt.PALETTE[1] if a > 0 else cplt.PALETTE[0] for a in along]
    else:
        w = np.asarray(weights)[m.numpy()]
        cols = cmap(w / (vmax or w.max()))
    for a, b, col in zip(along, across, cols):
        ax.add_patch(Circle((a, b), 0.5 * 0.55, facecolor=col, edgecolor=cplt.INK, lw=0.6))
    ax.plot([0], [0], marker="*", ms=13, color=cplt.PALETTE[5])
    ax.text(R_CUT * 0.55, R_CUT * 1.02, "ahead", color=cplt.PALETTE[1], ha="center")
    ax.text(-R_CUT * 0.55, R_CUT * 1.02, "behind", color=cplt.PALETTE[0], ha="center")
    ax.set_xlim(-R_CUT - 0.3, R_CUT + 0.3), ax.set_ylim(-0.15, R_CUT + 0.35), ax.set_aspect("equal")
    ax.set_xlabel(r"distance along $\hat{g}$"), ax.set_ylabel(r"distance across $\hat{g}$")
    if title:
        ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
for ax, i in zip(axes[:2], (0, 1)):
    n = int(train_raw["real"][i].sum())
    draw_neighbourhood(ax, train_raw, i, title=f"set {i}: {n} neighbours within $r_c$")
axes[2].hist(counts.numpy(), bins=np.arange(int(counts.min()) - 0.5, int(counts.max()) + 1.5), color=cplt.PALETTE[2])
axes[2].axvline(N_MAX, color=cplt.PALETTE[6], lw=2)
axes[2].text(N_MAX - 0.4, 0.97, f"N_MAX = {N_MAX}\n(padded length)", transform=axes[2].get_xaxis_transform(),
             ha="right", va="top", fontsize=8, color=cplt.PALETTE[6])
axes[2].set_xlim(right=N_MAX + 1)
axes[2].set_xlabel("real neighbours in the set"), axes[2].set_ylabel("number of sets")
axes[2].set_title("set sizes vary, so we pad", fontsize=10)
plt.tight_layout()
plt.show()
```

The left panels fold each neighbourhood into a half-plane around the query point (star): the horizontal axis is
the distance along the settling direction, the vertical axis the distance across it. Every neighbour on the right
half counts toward $y_{\text{ahead}}$, every neighbour on the left toward $y_{\text{behind}}$. The fold throws
away the azimuth (the angle *around* $\hat{\mathbf g}$), so two particles on opposite sides of the axis can land
on top of each other in the picture. The spheres do not overlap in 3D; only their shadows on this half-plane do.
The fold keeps exactly the two geometric inputs, $r_j$ and $c_j$.

### 3.1 The model

The network is deliberately minimal, so that the attention weights are the whole story:

1. One extra **query token** stands for the query point itself. It has features $(0, 0, 0)$ and a flag $1$; the
   neighbours have their three features and a flag $0$.
2. A small MLP embeds every token to width $D = 16$.
3. **One** multi-head self-attention layer (the course's `MultiHeadSelfAttention`) with the padding mask.
4. A linear readout of the query token's output gives $(\hat y_{\text{ahead}}, \hat y_{\text{behind}})$.

The query token's attention row is then literally "how the query point looks at its neighbours", one row per
head. There is no feed-forward layer after attention and the readout is linear, so the network cannot divide
one number by another after attention. A single head can still put indicator-weighted sums such as
$\sum_{j:\,c_j>0} v_j$ and the group sizes into different channels of its value vector, but all of them arrive
divided by the same total, and turning them into two group means needs a division the linear readout cannot do.
Each head's softmax performs that division itself. This is a deliberate choice that makes the role of the heads
easy to see; Exercise 5 adds a nonlinear readout and measures how much of the gap it closes.

```{code-cell} ipython3
class QueryTokenNet(torch.nn.Module):
    """[query token, neighbour tokens] -> MLP embedding -> one masked MHSA layer -> readout at the query token."""

    def __init__(self, width=16, heads=2, hidden=32):
        super().__init__()
        self.embed = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, width))
        self.attn = MultiHeadSelfAttention(width, heads)
        self.read = torch.nn.Linear(width, 2)

    @staticmethod
    def tokens(feats):
        B, N, _ = feats.shape
        query = torch.zeros(B, 1, 4, dtype=feats.dtype)
        query[..., 3] = 1.0                                               # the flag
        return torch.cat([query, torch.cat([feats, torch.zeros(B, N, 1, dtype=feats.dtype)], -1)], 1)

    def forward(self, feats, real, use_mask=True):
        B = feats.shape[0]
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool), real], 1)  # the query token is always real
        mask = keep[:, None, None, :] if use_mask else None              # (B, 1, 1, 1 + N): same for every head and row
        return self.read(self.attn(self.embed(self.tokens(feats)), mask=mask)[:, 0])

    def query_weights(self):
        """Attention of the query token over the neighbour slots, (B, H, N), from the last forward."""
        return self.attn.last_weights[:, :, 0, 1:]
```

Training is plain Adam on the mean squared error of the standardised targets, in float32:

```{code-cell} ipython3
Y_MEAN, Y_STD = train_raw["y"].mean(0), train_raw["y"].std(0)


def fit(heads, seed=0, epochs=60, lr=5e-3, batch=128, width=16, use_mask=True, net_cls=QueryTokenNet):
    torch.manual_seed(seed)
    net = net_cls(width, heads)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    f, real = train_f.float(), train_raw["real"]
    y = ((train_raw["y"] - Y_MEAN) / Y_STD).float()
    gen = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        perm, tot = torch.randperm(len(f), generator=gen), 0.0
        for i in range(0, len(f), batch):
            b = perm[i:i + batch]
            loss = ((net(f[b], real[b], use_mask=use_mask) - y[b]) ** 2).mean()
            opt.zero_grad(), loss.backward(), opt.step()
            tot += loss.item() * len(b)
        losses.append(tot / len(f))
    return net, losses


def evaluate(net, feats=None, real=None, use_mask=True):
    feats = test_f if feats is None else feats
    real = test_raw["real"] if real is None else real
    with torch.no_grad():
        p = net(feats.float(), real, use_mask=use_mask).double() * Y_STD + Y_MEAN
    return [mt.r2(p[:, i:i + 1], test_raw["y"][:, i:i + 1]) for i in range(2)]


nets, hist, scores = {}, {}, {}
for H in (1, 2):
    t = time.time()
    nets[H], hist[H] = fit(H)
    scores[H] = evaluate(nets[H])
    print(f"H = {H}: {param_count(nets[H]):,} parameters, test R2 ahead {scores[H][0]:.3f}, "
          f"behind {scores[H][1]:.3f}  ({time.time() - t:.0f} s)")
```

```{code-cell} ipython3
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
xpos = np.arange(2)
for k, H in enumerate((1, 2)):
    axes[0].bar(xpos + (k - 0.5) * 0.36, scores[H], width=0.34, color=cplt.PALETTE[k],
                label=f"{H} head{'s' if H > 1 else ''}, width 16")
axes[0].set_xticks(xpos, [r"$y_{\mathrm{ahead}}$", r"$y_{\mathrm{behind}}$"])
axes[0].set_ylabel(r"test $R^2$"), axes[0].set_ylim(0.5, 1.12)
axes[0].set_title("same parameters, same training", fontsize=10)
axes[0].legend(loc="upper center", ncol=2, fontsize=8)
cplt.learning_curves({f"{H} head{'s' if H > 1 else ''}": hist[H] for H in (1, 2)}, ax=axes[1],
                     title="training loss (standardised targets)")
axes[1].set_ylabel("train MSE")
plt.tight_layout()
plt.show()
```

The two networks have exactly the same parameters and the same training. The only difference is whether the
query token's 16 channels are scored with one softmax or with two. The single head ends up clearly worse on
both outputs. The loss curves tell the same story in time. Both models first drop onto the same plateau, the best
that one broad average with a linear readout can do. The two-head model then leaves it, and the next section shows why: that is when
its heads split the neighbourhood between them.

## 4. What the heads learned

Look at the query token's attention row in each head. For every test neighbour, plot its weight against its
angle $c_j$. To compare sets of different sizes, we multiply each weight by the set size $n$: a value of 1 means
"exactly the uniform share $1/n$", 0 means "ignored".

The head index that ends up looking ahead depends on the random initialisation, so we name the heads from what
they do: the head with the larger share of weight ahead is called the *ahead head*.

```{code-cell} ipython3
def head_shares(net, raw=None, feats=None):
    """Mean weight each head gives to the neighbours ahead and behind (query token row). Defaults: the current test set."""
    raw = test_raw if raw is None else raw
    feats = test_f if feats is None else feats
    with torch.no_grad():
        net(feats.float(), raw["real"])
    w = net.query_weights().double()                                   # (P, H, N)
    ahead = (raw["real"] & (raw["c"] > 0))[:, None]
    behind = (raw["real"] & (raw["c"] < 0))[:, None]
    return w, (w * ahead).sum(-1).mean(0), (w * behind).sum(-1).mean(0)


w1, a1, b1 = head_shares(nets[1])
w2, a2, b2 = head_shares(nets[2])
order = [int(a2.argmax()), int(a2.argmin())]                           # [ahead head, behind head]
names = {order[0]: "ahead head", order[1]: "behind head"}
print(f"1 head : weight on neighbours ahead {a1[0]:.2f}, behind {b1[0]:.2f}")
for i in order:
    print(f"2 heads, head {i + 1} ({names[i]}): weight ahead {a2[i]:.2f}, behind {b2[i]:.2f}")
assert order[0] != order[1]
# the prose below describes a successful split; stop the build if training did not produce one
assert a2[order[0]] > 0.8 and a2[order[1]] < 0.2 and min(scores[2]) > max(scores[1])
```

```{code-cell} ipython3
:tags: [hide-input]
real_t = test_raw["real"]
n_t = real_t.sum(1, keepdim=True).double()
c_all = test_raw["c"][real_t].numpy()
rel1 = (w1[:, 0] * n_t)[real_t].numpy()
rel2 = [(w2[:, i] * n_t)[real_t].numpy() for i in order]
c_t = test_raw["c"]
n_ahead_t = (real_t & (c_t > 0)).sum(1, keepdim=True).double().expand_as(c_t)
n_behind_t = (real_t & (c_t < 0)).sum(1, keepdim=True).double().expand_as(c_t)
n_half = torch.where(c_t > 0, n_ahead_t, n_behind_t)[real_t].numpy()    # size of the half each neighbour is in
n_over_half = float(np.median((n_t.expand_as(c_t)[real_t]).numpy() / n_half))
rng = np.random.default_rng(0)
sel = rng.choice(len(c_all), size=min(2500, len(c_all)), replace=False)

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), sharey=True)
panels = ((rel1, "1 head", cplt.PALETTE[3]),
          (rel2[0], f"2 heads: head {order[0] + 1} (ahead head)", cplt.PALETTE[1]),
          (rel2[1], f"2 heads: head {order[1] + 1} (behind head)", cplt.PALETTE[0]))
bins = np.linspace(-1, 1, 21)
mid = 0.5 * (bins[1:] + bins[:-1])
for ax, (rel, title, col) in zip(axes, panels):
    ax.scatter(c_all[sel], rel[sel], s=5, alpha=0.25, color=col, lw=0, label="one neighbour")
    idx = np.digitize(c_all, bins) - 1
    med = [np.median(rel[idx == b]) if np.any(idx == b) else np.nan for b in range(len(mid))]
    ax.plot(mid, med, color=cplt.INK, lw=2, label="median")
    ax.axhline(1.0, color=cplt.INK, lw=0.8, ls=":", label="1: uniform over all n")
    ax.axhline(n_over_half, color=cplt.PALETTE[5], lw=1.2, ls="--", label=r"$n/n_{\mathrm{half}}$: uniform over one half")
    ax.axvline(0.0, color=cplt.INK, lw=0.8)
    ax.set_xlabel(r"$c_j$  (behind $\leftarrow$  $\rightarrow$ ahead)"), ax.set_title(title, fontsize=10)
axes[0].set_ylabel(r"$n \times$ attention weight  (1 = uniform)")
axes[1].legend(loc="upper left", fontsize=7.5)
axes[0].set_ylim(-0.2, float(np.percentile(np.concatenate([rel1] + rel2), 99.5)) * 1.1)
plt.tight_layout()
plt.show()
```

The picture is a step function in each of the two heads. One head gives the neighbours ahead most of its weight
and almost nothing to those behind; the other head does the reverse. The single head (left) compromises with a
broad distribution. For a specialised head the right reference is not the dotted line ("uniform over all $n$")
but the dashed one: uniform over its own half, $n/n_{\text{half}}$, about 2. How close to that do the heads come?
Multiply each weight by the size of the half it falls in, $n_{\text{half}}$, so that 1 means "exactly the share a
mean over that half would give":

```{code-cell} ipython3
c_rt = test_raw["c"][real_t].numpy()
for k, (rel, side) in enumerate(((rel2[0], c_rt > 0), (rel2[1], c_rt < 0))):
    x = (rel / n_t.expand_as(test_raw["c"])[real_t].numpy() * n_half)[side]      # n_half x weight, own half
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    print(f"{('ahead', 'behind')[k]} head, neighbours in its own half: n_half x weight median {p50:.2f}, "
          f"10-90% range {p10:.2f} to {p90:.2f}")
```

So the specialised heads are close to, not exactly, uniform within their half: the middle 80% of neighbours get
between about 0.75 and 1.2 times the share a mean over the half would give. The bump near $|c_j|\approx 0.3$ in the scatter plot is where the
head is most confident that a neighbour is on its side. The leftover non-uniformity is part of why two heads
still do not reach $R^2 = 1$.

### 4.1 Why the heads split: a ramp of scores, then a softmax

The weights are the end of the story. The beginning is the score $s^{(h)}_{0j} = \mathbf Q^{(h)}_0\cdot
\mathbf K^{(h)}_j/\sqrt{d_h}$ of the query token (index 0) for neighbour $j$, in head $h$. The softmax ignores a
constant added to all the scores of one row, so we plot each score minus the mean score of its set, against $c_j$.
Below it, the same neighbours after the softmax:

```{code-cell} ipython3
:tags: [hide-input]
net = nets[2]
with torch.no_grad():
    tok = net.embed(net.tokens(test_f.float()))
    q2, k2, _ = (split_heads(t, 2) for t in net.attn.qkv(tok).chunk(3, dim=-1))
    s0 = (q2[:, :, :1] @ k2.transpose(-2, -1))[:, :, 0, 1:].double() / math.sqrt(q2.shape[-1])   # (P, H, N)
fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.6), sharex=True, layout="constrained")
for col, (hd, name, colour) in enumerate(zip(order, ("ahead head", "behind head"), (cplt.PALETTE[1], cplt.PALETTE[0]))):
    sc = s0[:, hd]
    sc = sc - (sc * real_t).sum(1, keepdim=True) / real_t.sum(1, keepdim=True)
    y_s = sc[real_t].numpy()
    axes[0, col].scatter(c_all[sel], y_s[sel], s=5, alpha=0.3, color=colour, lw=0)
    idx = np.digitize(c_all, bins) - 1
    axes[0, col].plot(mid, [np.median(y_s[idx == b]) if np.any(idx == b) else np.nan for b in range(len(mid))],
                      color=cplt.INK, lw=2)
    axes[0, col].axhline(0, color=cplt.INK, lw=0.8, ls=":"), axes[0, col].axvline(0, color=cplt.INK, lw=0.8)
    axes[0, col].set_title(f"{name}: score $s_{{0j}}$ minus its set mean", fontsize=10)
    axes[1, col].scatter(c_all[sel], rel2[col][sel], s=5, alpha=0.3, color=colour, lw=0)
    axes[1, col].axvline(0, color=cplt.INK, lw=0.8)
    axes[1, col].set_title(f"{name}: after the softmax ($n\\times$ weight)", fontsize=10)
    axes[1, col].set_xlabel(r"$c_j$  (behind $\leftarrow$  $\rightarrow$ ahead)")
axes[0, 0].set_ylabel("relative score"), axes[1, 0].set_ylabel(r"$n \times$ weight")
plt.show()
```

Each head has learned a score that is roughly flat on its own side and falls off steeply, almost linearly in
$c_j$, on the other side; the two heads are mirror images. The drop is ten or more units, and the softmax
exponentiates it: a neighbour scored 10 below the others gets $e^{-10}\approx 5\times10^{-5}$ times their weight.
So a smooth ramp in score space becomes a sharp step in weight space. In the language of Lesson 05b: in the ahead
head, the keys of neighbours behind cast short (strongly negative) shadows on the query token's query direction,
and in the behind head it is the keys of neighbours ahead. Nothing in the architecture had to be told where the step is; gradient descent put it at
$c_j \approx 0$ because that is where the target changes group.

The same thing, drawn on two neighbourhoods. Each particle is coloured by the weight the query point (star) gives
it, separately per head:

```{code-cell} ipython3
:tags: [hide-input]
# two test sets with plenty of neighbours and none sitting right on the dividing line
ok = (real_t.sum(1) >= 8) & ~((test_raw["c"].abs() < 0.2) & real_t).any(1)
show = torch.nonzero(ok)[:2, 0].tolist()
fig, axes = plt.subplots(2, 3, figsize=(12.4, 5.4), layout="constrained")
for row, i in enumerate(show):
    ws = [w1[i, 0]] + [w2[i, k] for k in order]
    vmax = float(max(w.max() for w in ws))
    titles = ["1 head", f"ahead head (head {order[0] + 1} of 2)", f"behind head (head {order[1] + 1} of 2)"]
    for ax, w, ttl in zip(axes[row], ws, titles):
        draw_neighbourhood(ax, test_raw, i, weights=w.numpy(), vmax=vmax, title=f"set {i}, {ttl}")
sm = plt.cm.ScalarMappable(cmap=cplt.CMAP_SEQUENTIAL, norm=plt.Normalize(0, 1))
fig.colorbar(sm, ax=axes, shrink=0.7, label="weight / largest weight in the figure row")
plt.show()
```

Two lessons from this small experiment carry over to real transformers.

- **Heads are a resource, not a guarantee.** Nothing in the architecture said "head 1, look ahead". The split
  emerged from training because the task rewards two differently normalised averages. With more heads than the task needs,
  some heads duplicate each other or stay broad (Exercise 1), and whether the split is found at all is an
  outcome of training, not a guarantee (Exercise 2).
- **More heads cost nothing in parameters**, but they shrink each head's dimension $d_h = D/H$. Every head's
  scores $\mathbf Q^{(h)}\cdot\mathbf K^{(h)}$ are built from only $d_h$ numbers, so very many heads of tiny
  dimension can each express only simple patterns.

## 5. Padding masks: sets of different sizes in one batch

### 5.1 What goes wrong without the mask

The batch tensor has `N_MAX` slots per set, and most slots in most sets are padding. Without a mask, a padded
slot is just another token: its zero features are embedded (to something that is not zero, because of the
biases), it gets a key and a value, and it receives attention weight like any real neighbour. There are two ways
to get this wrong, and we try both on the two-head model:

- **Train without the mask** from the start. This is the fair comparison: the network sees padding during
  training and can learn to live with it.
- **Train with the mask, then forget it at inference.** This mismatch is a common bug (for example a
  `mask=None` default in an evaluation script), and it is far more damaging.

```{code-cell} ipython3
t = time.time()
net_nm, _ = fit(2, use_mask=False)                                      # trained with padding visible
r2_mask = evaluate(nets[2])
r2_nm = evaluate(net_nm, use_mask=False)
r2_forgot = evaluate(nets[2], use_mask=False)
with torch.no_grad():
    net_nm(test_f.float(), real_t, use_mask=False)
leak = (net_nm.query_weights() * (~real_t)[:, None]).sum(-1)            # weight on padded slots, (P, H)
print(f"trained and tested with mask        : test R2 ahead {r2_mask[0]:.3f}, behind {r2_mask[1]:.3f}")
print(f"trained and tested without mask     : test R2 ahead {r2_nm[0]:.3f}, behind {r2_nm[1]:.3f}  "
      f"({time.time() - t:.0f} s to train)")
print(f"trained with mask, tested without it: test R2 ahead {r2_forgot[0]:.3f}, behind {r2_forgot[1]:.3f}")
print(f"unmasked network: the query token puts on average {leak.mean():.2f} of its weight on padding "
      f"(largest {leak.max():.2f})")
assert max(r2_nm) < min(r2_mask) and max(r2_forgot) < min(r2_mask)
```

The network trained without a mask is clearly worse than the masked one, though not catastrophically: with zeros
in every padded slot, the padding is a consistent signal, and the number of padded slots even tells the network
how many real neighbours there are. Forgetting the mask on a network that never saw padding is much worse.

The deeper problem with the unmasked network is that its predictions depend on what we chose to write into the
padded slots, a choice that has nothing to do with the physics. The same network, the same real neighbours,
three different fill values:

```{code-cell} ipython3
def fill_padding(feats, real, value):
    return torch.where(real[..., None], feats, torch.full_like(feats, value))

with torch.no_grad():
    for fill in (0.0, 1.0, -3.0):
        p = net_nm(fill_padding(test_f, real_t, fill).float(), real_t, use_mask=False)
        r2_fill = [mt.r2(p.double()[:, i:i + 1] * Y_STD[i] + Y_MEAN[i], test_raw["y"][:, i:i + 1]) for i in range(2)]
        print(f"unmasked net, padded slots filled with {fill:+.0f}: first prediction "
              f"{(p[0].double() * Y_STD + Y_MEAN).numpy().round(4)}, test R2 {np.round(r2_fill, 3)}")
```

A model whose answer changes when you edit slots that hold no particle is not a model of the particles. The mask
removes this dependence exactly, as the next section proves.

### 5.2 The mask, and why it is exact

A boolean mask $m_{jk}\in\{\text{True},\text{False}\}$ (True = "may attend") is applied to the logits before the
softmax:

$$
\tilde z_{jk} = \begin{cases} z_{jk} & m_{jk} = \text{True} \\ -\infty & m_{jk} = \text{False}\end{cases},
\qquad w_{jk} = \frac{e^{\tilde z_{jk}}}{\sum_{k'} e^{\tilde z_{jk'}}} .
$$

Since $e^{-\infty} = 0$ **exactly** in floating point, a masked key contributes exactly zero to the numerator and
to the denominator. The real keys renormalise among themselves, as if the padded slots had never existed. Then
$\mathbf a_j = \sum_k w_{jk}\mathbf V_k$ contains the padded values only as $0\cdot\mathbf V_k$, which is exactly
zero for any finite $\mathbf V_k$. These are not approximations, so we can check them with `==`, not with a
tolerance.

For a padding mask, $m_{jk}$ depends only on the key $k$: every row and every head uses the same mask. That is why
`QueryTokenNet` passes a mask of shape `(B, 1, 1, 1 + N)`, which broadcasts over heads and rows.

**Claim 1: padded slots get weight exactly zero.**

```{code-cell} ipython3
net64 = copy.deepcopy(nets[2]).double().eval()
f64, real64 = test_f.clone(), test_raw["real"]
with torch.no_grad():
    out_ref = net64(f64, real64)
W_all = net64.attn.last_weights                                         # (P, H, 1 + N, 1 + N), every row
keep = torch.cat([torch.ones(len(real64), 1, dtype=torch.bool), real64], 1)
pad_w = W_all.masked_select(~keep[:, None, None, :].expand_as(W_all))
print(f"{pad_w.numel():,} weights on padded keys (all heads, all rows); "
      f"number that are not exactly 0.0: {int((pad_w != 0).sum())}")
row_sums = W_all.sum(-1)
print(report("rows still sum to one over the real keys", float((row_sums - 1).abs().max()), TOL))
assert int((pad_w != 0).sum()) == 0
```

**Claim 2: the output does not depend on the padded values, not even in the last bit.** We overwrite every padded
slot with large random numbers and compare the outputs for exact equality:

```{code-cell} ipython3
gpad = torch.Generator().manual_seed(1)
garbage = 1e3 * torch.randn(f64.shape, generator=gpad, dtype=torch.float64)
f_garbage = torch.where(real64[..., None], f64, garbage)
with torch.no_grad():
    out_garbage = net64(f_garbage, real64)
diff = float((out_garbage - out_ref).abs().max())
print(f"padded features changed by up to {garbage.abs().max():.0f}; largest output change: {diff}")
assert diff == 0.0
```

**Claim 3: the gradient with respect to the padded inputs is exactly zero**, so the padding cannot leak into
training either:

```{code-cell} ipython3
f_req = f_garbage.clone().requires_grad_(True)
net64(f_req, real64).sum().backward()
g_pad = float(f_req.grad[~real64].abs().max())
g_real = float(f_req.grad[real64].abs().max())
print(f"largest |d output / d input|: padded slots {g_pad}, real slots {g_real:.2e}")
assert g_pad == 0.0 and g_real > 0
```

**Claim 4: padding plus mask equals no padding at all.** Each set, cut to its real neighbours and run on its own
with no mask, must give the same prediction as its row in the padded batch. Here the shapes differ, so the
library may add up the same numbers in a different order, and we compare to round-off:

```{code-cell} ipython3
worst = 0.0
with torch.no_grad():
    for i in range(200):
        n = int(real64[i].sum())
        alone = net64(f64[i:i + 1, :n], torch.ones(1, n, dtype=torch.bool), use_mask=False)
        worst = max(worst, float((alone - out_ref[i]).abs().max()))
print(report("padded + mask vs each set alone (200 sets)", worst, TOL))
```

**Claim 5: where the padding sits does not matter.** Shuffling the slots of every set, real and padded together,
with the mask shuffled the same way, must leave the output unchanged. This is the permutation invariance of
Lesson 05d, now with padding:

```{code-cell} ipython3
check_permutation(lambda f, m: net64(f, m), (f64[:64], real64[:64]), axis=1, mode="invariant",
                  name="masked net, slots shuffled with their mask");
```

Our hand-written `attention` must agree with PyTorch's `F.scaled_dot_product_attention`, which uses the same
convention (True = may attend) for a boolean `attn_mask`. In float64 on the CPU it runs its plain math path, not a
fused GPU kernel (Lesson 05b, Section 7):

```{code-cell} ipython3
with torch.no_grad():
    tok = net64.embed(net64.tokens(f64[:64]))
    q, k, v = (split_heads(t, 2) for t in net64.attn.qkv(tok).chunk(3, dim=-1))
    m = keep[:64, None, None, :]
    ours = attention(q, k, v, mask=m)
    fused = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
print(report("course attention vs F.scaled_dot_product_attention, bool mask", float((ours - fused).abs().max()), TOL))
```

:::{admonition} Watch the convention
:class: warning
`torch.nn.MultiheadAttention` takes its padding mask as `key_padding_mask`, where **True means "ignore this
key"**, the opposite of `attn_mask` in `scaled_dot_product_attention` and of the course's helpers. Passing the
mask of real tokens there silently masks the real tokens and keeps the padding.
:::

```{code-cell} ipython3
lengths = torch.tensor([3, 7, 5])
real_mask = padding_mask(lengths, L_)                                    # (B, 1, 1, L), True = real
with torch.no_grad():
    ref, _ = mha_inline(h, Wq, Wk, Wv, Wo, H_, mask=real_mask)
    right, _ = th_mha(h, h, h, key_padding_mask=~real_mask[:, 0, 0], need_weights=False)
    wrong, _ = th_mha(h, h, h, key_padding_mask=real_mask[:, 0, 0], need_weights=False)
print(report("nn.MultiheadAttention with key_padding_mask = ~real", float((right - ref).abs().max()), TOL))
print(f"nn.MultiheadAttention with key_padding_mask = real: max error {(wrong - ref).abs().max():.2f}  (wrong!)")
```

### 5.3 Two ways to break an exact mask

The proofs above have two small print conditions. Both have caught real code.

**The padded values must be finite.** The masked weight is exactly $0$, but $0\cdot\text{NaN} = \text{NaN}$ and
$0\cdot\infty = \text{NaN}$. A NaN left in a padded slot (for example from normalising an empty slot by its
zero length) spreads to every output:

```{code-cell} ipython3
f_nan = torch.where(real64[..., None], f64, torch.full_like(f64, float("nan")))
with torch.no_grad():
    out_nan = net64(f_nan, real64)
    out_clean = net64(torch.where(real64[..., None], f_nan, torch.zeros_like(f_nan)), real64)
print("NaN in padded slots      -> outputs that are NaN:", int(out_nan.isnan().any(-1).sum()), "of", len(out_nan))
print("same, zeroed before use  -> outputs that are NaN:", int(out_clean.isnan().any(-1).sum()),
      "| equal to the reference:", torch.equal(out_clean, out_ref))
```

The fix is the line we already had in `features`: overwrite padded slots with a finite value (zero) before they
enter the network.

**No row may be fully masked.** If every key in a row is masked, the softmax computes $-\infty - (-\infty)$,
which is NaN. With a cutoff radius, a query point with no neighbour inside $r_c$ produces exactly such a row.
Our network never hits this, because the query token is always real, so every row has at least one key:

```{code-cell} ipython3
qe = torch.randn(1, 1, 4, 8, dtype=torch.float64)
print("attention over a row with every key masked:",
      attention(qe, qe, qe, mask=torch.zeros(1, 1, 1, 4, dtype=torch.bool))[0, 0, 0, :3])
empty = torch.zeros(1, N_MAX, dtype=torch.bool)                          # a query point with no neighbours
with torch.no_grad():
    print("QueryTokenNet on a set with zero neighbours:", net64(torch.zeros(1, N_MAX, 3, dtype=torch.float64), empty))
```

One more consequence of the mask: the rows of the *padded* tokens (padded slots acting as queries) are still
computed, and their outputs are meaningless. That is harmless as long as nothing reads them. Our readout uses
only the query token. A model that pools over all tokens must pool with the mask too, for example a mean over
the real tokens only (Exercise 3).

## 6. Additive float masks: the general form

A boolean mask is a special case of something more flexible: a real number $b_{jk}$ added to the logits,

$$
w_{jk} = \operatorname{softmax}_k\big(z_{jk} + b_{jk}\big)
       = \frac{e^{z_{jk}}\,e^{b_{jk}}}{\sum_{k'} e^{z_{jk'}}\,e^{b_{jk'}}} .
$$

So adding $b_{jk}$ multiplies the unnormalised weight of key $k$ by $e^{b_{jk}}$, and the softmax renormalises.
A bias of $+1$ makes a key $e$ times more attractive, $-2$ about seven times less, and $-\infty$ removes it.
The boolean mask is the limit $b_{jk}\to-\infty$ on the masked entries. PyTorch accepts both forms in the same
`attn_mask` argument: a bool tensor is a mask, a float tensor is added. The course's `attention` keeps them as two
arguments, `mask` and `bias`.

```{code-cell} ipython3
gz = torch.Generator().manual_seed(2)
z = torch.randn(4, 9, generator=gz, dtype=torch.float64)
b = torch.randn(4, 9, generator=gz, dtype=torch.float64)
lhs = softmax(z + b)
rhs = softmax(z) * b.exp()
rhs = rhs / rhs.sum(-1, keepdim=True)
print(report("softmax(z + b) == softmax(z) e^b, renormalised", float((lhs - rhs).abs().max()), TOL))

with torch.no_grad():
    bias_inf = torch.zeros(m.shape, dtype=torch.float64).masked_fill(~m, float("-inf"))
    via_bias = attention(q, k, v, bias=bias_inf)
print("bias of -inf on the padded keys == boolean mask, bit for bit:", torch.equal(via_bias, ours))
```

A *finite* large negative bias is a soft mask: it leaves a tiny weight on the masked keys. How tiny depends on
the size of the bias and on the floating-point format:

```{code-cell} ipython3
:tags: [hide-input]
mags = np.logspace(0, np.log10(2000), 400)                             # bias sizes 1 ... 2000
z_logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
fig, ax = plt.subplots(figsize=(6.8, 3.8))
styles = {torch.float64: dict(color=cplt.PALETTE[0], ls="-", lw=2.5, zorder=2),
          torch.float32: dict(color=cplt.PALETTE[1], ls="--", lw=1.8, zorder=3)}
for dt in (torch.float64, torch.float32):
    leaked = []
    for s_ in mags:
        bb = torch.zeros(m.shape, dtype=dt).masked_fill(~m, -float(s_))
        w_ = softmax(z_logits.to(dt) + bb)
        leaked.append(float((w_ * (~m).to(dt)).sum(-1).max()))
    leaked = np.array(leaked)
    name = str(dt).split(".")[-1]
    ax.loglog(mags, np.where(leaked > 0, leaked, np.nan), label=name, **styles[dt])
    first_zero = mags[np.argmax(leaked == 0)] if (leaked == 0).any() else None
    if first_zero is not None:
        ax.axvline(first_zero, color=styles[dt]["color"], ls=":", lw=1.4)
        ax.text(first_zero * 1.08, 1e-200 if dt == torch.float64 else 1e-150, f"{name}:\nexactly 0\nfrom b = {first_zero:.0f}",
                color=styles[dt]["color"], fontsize=8, va="center")
guide = np.exp(-mags)
ax.loglog(mags, np.where(guide > 0, guide, np.nan), color=cplt.INK, lw=1.0, ls=":", label=r"$e^{-b}$ (guide)", zorder=4)
ax.set_ylim(1e-320, 10.0)
ax.set_yticks([10.0 ** e for e in range(0, -321, -40)])
ax.set_xlabel("size $b$ of the negative bias on padded keys")
ax.set_ylabel("largest total weight left on padding")
ax.set_title("a finite bias is a soft mask", fontsize=10)
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()
```

The two formats agree until float32 runs out of range, and both follow $e^{-b}$ up to a factor of order one (on
this scale the dotted guide disappears under the curves): each unit of negative bias divides the leaked weight by
$e$, exactly as the formula above says. A bias of a few tens already removes a key for every
practical purpose, and beyond the underflow point of $e^{x}$ the weight is exactly zero. That point comes much
earlier in float32 ($e^{x}$ underflows near $x\approx-104$) than in float64 (near $x\approx-745$); the
dotted lines are shifted from these values by the scores themselves, which are added to the bias.
Prefer the boolean mask anyway: it says what you mean, and it cannot be undone by a large logit.

The real use of float biases is **soft, learned preferences**. In model B, a small network maps the geometry of
each pair of neighbours $(j,k)$, using only distances and dot products, to one bias $b^{(\ell,h)}_{jk}$ for
every layer $\ell$ and every head $h$ (so `depth * heads` bias channels), and adds it to the logits exactly as
above. Neighbour $j$ can then attend more to a neighbour $k$ that sits, say, between $j$ and the query point. Lesson [05f](05f_invariant_attention.md) builds this, and it is the way the
model represents shielding. The research model works with a fixed number of nearest neighbours, so every set
has the same size and it needs no padding mask at all; padding appears as soon as neighbourhoods are defined by a
cutoff radius, as in this lesson.

```{figure} ../_static/media/S5_attention/S5_attention.png
:width: 100%
:alt: Two panels. Left, a plain sum weights neighbours A, B and C equally although B sits in A's shadow. Right, attention with a pair bias b(P_BA) gives B, in A's shadow, a small effective weight.

What a learned pair bias buys, previewed from Lessons 05e and 05f. A plain sum (left, model A0) gives
neighbour B the same weight whether or not neighbour A is in the way. In attention (right, model B), a bias
$b(P_{BA})$ computed from the geometry of the pair (B, A) is added to the logits exactly as in this section, and
lowers B's weight relative to C's because B sits in A's shadow. The still is drawn on a dark background in both
page themes. The full animation S5, with an interactive 3D version, plays in
[Lesson 05a, Section 3](05a_shielding.md); it writes $i$ for the token that looks and $j$ for the token looked at.
```

## 7. Causal masks, and why particles do not use them

Language models generate text one token at a time, and token $j$ must not see the tokens after it. Their
**causal mask** allows $m_{jk} = \text{True}$ only for $k \le j$, a lower triangle:

```{code-cell} ipython3
print(causal_mask(6).int())
```

For particles this mask is wrong, and the reason is the theme of Lesson 05d. The neighbours of a point form a
**set**: the slot a particle lands in (here: its rank by distance) is a bookkeeping choice. A causal mask turns
that choice into physics. The nearest neighbour could not attend to anyone else, and relabelling the
neighbours would change the answer. The permutation check catches it:

```{code-cell} ipython3
torch.manual_seed(0)
mhsa = MultiHeadSelfAttention(8, 2).double()
hh = torch.randn(4, 6, 8, dtype=torch.float64)
check_permutation(lambda x: mhsa(x), (hh,), axis=1, mode="equivariant", name="MHSA, no mask")
try:
    check_permutation(lambda x: mhsa(x, mask=causal_mask(6)), (hh,), axis=1, mode="equivariant",
                      name="MHSA with a causal mask")
except AssertionError as err:
    print("caught:", err)
```

A padding mask passes the same check (Claim 5) because it is attached to the tokens and moves with them when they
are relabelled. A causal mask is attached to the *slots*, so it does not.

```{code-cell} ipython3
print(f"lesson runtime so far: {time.time() - T0:.0f} s")
```

## Summary

- **Multi-head attention** splits the width $D$ into $H$ blocks of $d_h = D/H$ channels, runs one softmax per
  block, and concatenates the results before a final linear map. It is exactly $H$ independent single-head
  attentions, with the same parameters as one head of width $D$. We checked this against a loop over heads,
  the course module and `nn.MultiheadAttention`.
- One head yields one probability distribution per token, so every channel it delivers is divided by the same
  total. It cannot normalise one quantity by the size of group A and another by the size of group B. On the
  ahead/behind toy, with a linear readout, two heads at the same parameter count **specialise**: a ramp of scores
  in $c_j$, sharpened by the softmax, gives each head one half of the neighbourhood with roughly uniform weight,
  and the single head falls behind.
- A **padding mask** sets masked logits to $-\infty$, so padded keys get weight **exactly zero**. The output and
  its gradient are then independent of the padded values (checked with `==`), and the result equals the
  unpadded computation to round-off. The fine print: padded values must be finite, and no row may be fully
  masked.
- An **additive float mask** multiplies each key's weight by $e^{b_{jk}}$. The boolean mask is its $-\infty$
  limit. Model B uses a learned float bias to inject pair geometry.
- A **causal mask** imposes an order on the tokens and breaks permutation symmetry. Particles have no order, so
  particle models do not use it.

## Exercises

1. **More heads than the task needs.** Train `fit(4)` and print `head_shares` for each of the four heads. Classify
   each head as ahead, behind or broad. Is the test $R^2$ better than with two heads?
2. **Is the specialisation robust?** Train the two-head model with seeds 0 to 3 and print each run's test $R^2$
   and the heads' weights ahead. Do all four runs split? Then look at a run that has *not* split: train the same
   four seeds for only `epochs=10`, while the loss curve of Section 3.1 is still on its plateau. What are the
   heads' weights ahead, and how does the $R^2$ compare with the single head's?
3. **Pooling with padding.** Replace the query-token readout by the mean of the outputs of all *real* neighbour
   tokens (no query token). Write the masked mean, then show numerically that an unmasked mean over all
   `N_MAX` slots makes the output depend on the padded values even when attention itself is masked.
4. **Masking by bias.** Repeat Claim 2 with `bias = -1e4` on padded keys instead of a boolean mask, once in
   float64 and once in float32. Is the output change still exactly zero? Use the figure of Section 6 to explain
   the answer.
5. **How much does a nonlinear readout help one head?** Subclass `QueryTokenNet` so that `self.read` is a small
   MLP (`Linear(16, 32)`, GELU, `Linear(32, 2)`) and train it with `fit(1, net_cls=...)` and `fit(2, net_cls=...)`.
   Compare the test $R^2$ with the linear-readout results of Section 3.1. What can the MLP do that the linear
   readout cannot, and why does it still not catch up with two heads?

:::{admonition} Solution 1
:class: dropdown
```python
net4, _ = fit(4)
print("test R2:", evaluate(net4))
_, a4, b4 = head_shares(net4)
for i in range(4):
    kind = "ahead" if a4[i] > 0.8 else "behind" if b4[i] > 0.8 else "broad"
    print(f"head {i + 1}: ahead {a4[i]:.2f}, behind {b4[i]:.2f} -> {kind}")
```
Typically at least one head looks ahead and at least one behind, and the others either duplicate them or stay
broad. The test $R^2$ is about the same as with two heads, at most slightly better: the task needs two averages,
so the extra heads have little new to contribute, while each head now works with only $d_h = 4$ channels. Heads are capacity that
training may or may not use.
:::

:::{admonition} Solution 2
:class: dropdown
```python
for seed in range(4):
    n_, _ = fit(2, seed=seed)
    _, a_, b_ = head_shares(n_)
    print(f"seed {seed}: test R2 {np.round(evaluate(n_), 3)}, weight ahead per head {a_.numpy().round(2)}")
```
In our runs, all four seeds split cleanly at 60 epochs: one head puts more than 0.85 of its weight ahead, the
other less than 0.15, and the test $R^2$ is about 0.97. The split is robust here, but it is not instantaneous.
```python
for seed in range(4):
    n_, _ = fit(2, seed=seed, epochs=10)
    _, a_, b_ = head_shares(n_)
    print(f"seed {seed}, 10 epochs: test R2 {np.round(evaluate(n_), 3)}, weight ahead per head {a_.numpy().round(2)}")
```
After 10 epochs no seed has split yet: both heads put between about 0.3 and 0.6 of their weight ahead, and the
test $R^2$ (about 0.83 ahead, 0.78 behind) is no better than the single head's. This is what an unsplit run looks
like, and it is the plateau in the loss curve of Section 3.1: two heads that both average broadly behave like one
head. Leaving the plateau is an outcome of optimisation, not a property the architecture guarantees. On this toy
we did not find a simple change that keeps a seed on the plateau for all 60 epochs, but in harder problems a run
can stay there much longer, so always check what the heads learned instead of assuming it.
:::

:::{admonition} Solution 3
:class: dropdown
```python
def masked_mean(x, real):                      # x (B, N, C), real (B, N)
    w = real[..., None].to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1)

class MeanPoolNet(QueryTokenNet):
    def forward(self, feats, real, masked_pool=True):
        tok = self.tokens(feats)[:, 1:]                              # neighbour tokens only
        h = self.attn(self.embed(tok), mask=real[:, None, None, :])
        pooled = masked_mean(h, real) if masked_pool else h.mean(1)
        return self.read(pooled)

torch.manual_seed(0)
mp = MeanPoolNet(16, 2).double()
for pool in (True, False):
    with torch.no_grad():
        d = (mp(f_garbage, real64, pool) - mp(f64, real64, pool)).abs().max().item()
    print(f"masked pool = {pool}: output change when padding changes = {d}")
```
With the masked mean the change is exactly zero. With a plain mean it is clearly not zero: attention never looked at the
padded keys, but the padded tokens' own rows (padded slots acting as queries) still produce outputs, and the
plain mean adds them in. Every reduction over tokens needs the mask, not only the softmax.
:::

:::{admonition} Solution 4
:class: dropdown
```python
def out_with_bias(net, feats, real, dtype):
    n_ = copy.deepcopy(net).to(dtype)
    B = len(feats)
    keep = torch.cat([torch.ones(B, 1, dtype=torch.bool), real], 1)
    bias = torch.zeros(B, 1, 1, keep.shape[1], dtype=dtype).masked_fill(~keep[:, None, None, :], -1e4)
    with torch.no_grad():
        q, k, v = (split_heads(t, 2) for t in n_.attn.qkv(n_.embed(n_.tokens(feats.to(dtype)))).chunk(3, -1))
        a = merge_heads(attention(q, k, v, bias=bias))
        return n_.read(n_.attn.out(a)[:, 0])

for dt in (torch.float64, torch.float32):
    d = (out_with_bias(nets[2], f_garbage, real64, dt) - out_with_bias(nets[2], f64, real64, dt)).abs().max()
    print(dt, float(d))
```
Both changes are exactly zero. A bias of $-10^4$ is far past the point where $e^{x}$ underflows to zero (about
$-745$ in float64 and about $-104$ in float32), so the masked weights are exactly zero, as in the figure. With
$-10$ instead, the padded keys would keep a small weight and the change would no longer be zero. Large garbage
values can also push a padded logit up by more than the bias pushes it down, which is one more reason to prefer
the boolean mask.
:::

:::{admonition} Solution 5
:class: dropdown
```python
class MLPReadoutNet(QueryTokenNet):
    def __init__(self, width=16, heads=2, hidden=32):
        super().__init__(width, heads, hidden)
        self.read = torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.GELU(), torch.nn.Linear(32, 2))

for H in (1, 2):
    n_, _ = fit(H, net_cls=MLPReadoutNet)
    print(f"MLP readout, {H} head(s): {param_count(n_):,} parameters, test R2 {np.round(evaluate(n_), 3)}")
```
With the MLP readout, one head gets much closer (test $R^2$ about 0.93 ahead and 0.92 behind, against 0.84 and
0.83 with the linear readout), and two heads improve a little too (about 0.96 and 0.97). The single head can
place indicator-weighted sums and group sizes in different channels, all divided by the same total; an MLP can
then approximate the ratio of two such channels, which is the per-group normalisation the linear readout could
not do. It only approximates it, from 16 noisy channels, while a second head's softmax performs that division
exactly. A feed-forward layer and more heads are complementary, which is why every transformer block has both.
:::

## References

- Multi-head attention and masking in the original transformer: {cite}`vaswani2017attention`.
- Attention over sets, including pooling by attention with a learned query (the idea behind our query token):
  {cite}`lee2019settransformer`.
- PyTorch documentation for
  [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
  (`attn_mask`: True = attend, or a float bias) and
  [`nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
  (`key_padding_mask`: True = ignore).
- MolSSI e3nn course {cite}`molssi_e3nn_course`, for the symmetry background used from Lesson 05f on:
  <https://molssi-ai.github.io/e3nn-course/book/intro.html>.
