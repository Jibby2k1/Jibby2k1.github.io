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

# One-particle disturbances and superposition

:::{admonition} What you will learn
:class: tip
1. The exact Stokes flow around one translating sphere: a $1/r$ *Stokeslet* plus a $1/r^3$ *dipole*, and
   what the toy's screening factor does to it. You will check no-slip, incompressibility and the Stokes
   equations numerically.
2. Why the one-sphere field lies in $\mathrm{span}(\mathbf x, \mathbf s)$, transforms as a vector under
   $O(3)$ and is fore-aft symmetric. A learned kernel must keep the first two properties and may break the
   third.
3. Why the flow of many spheres is a *sum* of one-sphere fields (linearity), and which velocity each sphere
   has to carry in that sum.
4. How large the error of naive superposition with the particles' own velocities is, as a function of
   separation and volume fraction, where on the slice it sits, and how much of it is a collective drift
   of the whole suspension rather than local crowding.
5. Why this motivates two things in the surrogate: a per-particle state found self-consistently (Part III),
   and a learned kernel in place of the Stokes formula (Part II).
:::

**Prerequisites:** [Lesson 01a](01a_settling_particles.md) (the toy snapshot and the surface-average
identity). Vectors, rotations and reflections are used as in the
[MolSSI e3nn course](https://molssi-ai.github.io/e3nn-course/book/intro.html).

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
import time
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt
import torch
from course_utils import plotting as cp
from course_utils import toy_data as td
from course_utils.toy_data import ToyConfig, ToySnapshot, make_snapshot, stokes_sphere_matrix
from course_utils.equivariance import check_equivariance, span_residual, report

torch.set_num_threads(min(4, torch.get_num_threads()))
T_START = time.time()
cfg = ToyConfig()                      # the course defaults: a = 0.5, screening length L = 1
a, Lsc = cfg.radius, cfg.screening_length
```

## 1. The flow around one sphere

### 1.1 The formula

A sphere of radius $a$ moves with velocity $\mathbf s$ through fluid that is at rest far away. The motion is
slow, so inertia is negligible and the fluid obeys the **Stokes equations** (viscosity $\mu$, pressure $p$):

$$
\mu \nabla^2 \mathbf u = \nabla p, \qquad \nabla\cdot\mathbf u = 0 .
$$

With no-slip on the sphere ($\mathbf u = \mathbf s$ on $|\mathbf x| = a$) the solution is classical
{cite}`happel1983low`. Write $\mathbf x$ for the offset from the centre, $r = |\mathbf x|$ and
$\hat{\mathbf x} = \mathbf x / r$:

$$
\mathbf u(\mathbf x) = G(\mathbf x)\,\mathbf s, \qquad
G(\mathbf x) = \underbrace{\frac{3a}{4r}\bigl(I + \hat{\mathbf x}\hat{\mathbf x}^{\mathsf T}\bigr)}_{\text{Stokeslet}}
\;+\; \underbrace{\frac{a^3}{4r^3}\bigl(I - 3\hat{\mathbf x}\hat{\mathbf x}^{\mathsf T}\bigr)}_{\text{potential dipole}},
\qquad p = \frac{3\mu a}{2}\,\frac{\mathbf s\cdot\mathbf x}{r^3}.
$$

The two pieces have distinct jobs:

- the **Stokeslet** is the flow made by a point force $\mathbf F = 6\pi\mu a\,\mathbf s$ (Stokes drag). It
  decays slowly, like $1/r$, and carries all the pressure;
- the **dipole** decays like $1/r^3$. It is a potential flow (no pressure, no vorticity) and its only job is
  to make the sum equal $\mathbf s$ exactly on the surface: at $r = a$ the two
  $\hat{\mathbf x}\hat{\mathbf x}^{\mathsf T}$ terms cancel and the two $I$ terms add to $I$.

The toy multiplies $G$ by a **screening factor** $e^{-(r-a)/L}$ (default $L = 1$). It equals 1 on the
surface, so no-slip is unchanged, and it makes each disturbance die within a few diameters. It stands in
crudely for the way a crowd of particles damps a disturbance, and it keeps sums over many particles bounded
in the toy's open box. We will see in Section 1.3 what it costs.

We implement $G\mathbf s$ in PyTorch, with the same symbols, and compare it with the toy's NumPy version.
The torch version lets us take exact derivatives with autograd below.

```{code-cell} ipython3
def stokes_u(x, s, a=0.5, L=None):
    """u = G(x) s for one sphere; x, s (..., 3) torch tensors. Stokeslet + dipole, optional screening."""
    r = x.norm(dim=-1, keepdim=True)
    xh = x / r
    xs = (xh * s).sum(-1, keepdim=True)                       # x^ . s
    stokeslet = 3 * a / (4 * r) * (s + xh * xs)
    dipole = a**3 / (4 * r**3) * (s - 3 * xh * xs)
    u = stokeslet + dipole
    return u * torch.exp(-(r - a) / L) if L is not None else u

rng = np.random.default_rng(0)
x_np = rng.normal(size=(500, 3)) * 2.0
x_np = x_np[np.linalg.norm(x_np, axis=1) > a]                 # outside the sphere
s_np = np.array([0.3, -1.0, 0.2])
for L in (None, Lsc):
    ours = stokes_u(torch.as_tensor(x_np), torch.as_tensor(s_np), a, L).numpy()
    toy = td.stokes_sphere_velocity(x_np, s_np, a, L)
    print(report(f"torch stokes_u == toy stokes_sphere_velocity (L = {L})", np.abs(ours - toy).max(), 1e-12))
```

### 1.2 What it looks like

A single sphere in the toy is a snapshot with one particle, no backflow, and slip equal to its velocity. We
draw the plane $z = 4$ through its centre, with the sphere moving down ($-y$). The same flow looks different
in two frames: in the **lab frame** the fluid far away is at rest and the sphere drags its neighbourhood
along; in the **sphere frame** (subtract $\mathbf s$) a uniform stream flows up past a sphere at rest. We
also split the lab-frame field into its two pieces.

```{code-cell} ipython3
cfg1 = replace(cfg, n_particles=1, backflow=False)
centre = np.full((1, 3), cfg.box / 2)
s1 = np.array([[0.0, -1.0, 0.0]])
one = ToySnapshot(positions=centre, velocity=s1, slip=s1.copy(), backflow=np.zeros(3),
                  phi_local=np.zeros(1), config=cfg1)
one_unscreened = ToySnapshot(centre, s1, s1.copy(), np.zeros(3), np.zeros(1), replace(cfg1, screening_length=None))

pts_c = np.random.default_rng(1).normal(size=(200, 3)) * 1.5 + centre
pts_c = pts_c[one.exterior_mask(pts_c)]
direct = td.stokes_sphere_velocity(pts_c - centre, s1[0], a, Lsc)
print(report("consistency (same construction): one-sphere snapshot field == G(x - r) s",
             np.abs(one.field(pts_c) - direct).max(), 1e-12))
```

```{code-cell} ipython3
:tags: [hide-input]
def pieces(p, which):
    """Stokeslet or dipole part of the unscreened one-sphere field at points p."""
    x = torch.as_tensor(p - centre)
    s = torch.as_tensor(s1[0])
    r = x.norm(dim=-1, keepdim=True); xh = x / r; xs = (xh * s).sum(-1, keepdim=True)
    out = 3 * a / (4 * r) * (s + xh * xs) if which == "stokeslet" else a**3 / (4 * r**3) * (s - 3 * xh * xs)
    return out.numpy()

zoom = (1.5, 6.5)
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), layout="constrained")
panels = [
    (one_unscreened, lambda p: one_unscreened.field(p), "lab frame: fluid dragged along"),
    (one_unscreened, lambda p: one_unscreened.field(p) - s1[0], "sphere frame: stream past a sphere"),
    (one_unscreened, lambda p: pieces(p, "stokeslet"), "Stokeslet part ($1/r$)"),
    (one_unscreened, lambda p: pieces(p, "dipole"), "dipole part ($1/r^3$)"),
]
for ax, (sn, fn, title) in zip(axes, panels):
    cp.field_slice(sn, field_fn=fn, ax=ax, n=121, streamlines=True, vmax=1.5, colorbar=False, title=title)
    ax.set_xlim(*zoom), ax.set_ylim(*zoom)
fig.colorbar(axes[0].images[0], ax=axes, shrink=0.85, label=r"in-plane speed $/\,|\mathbf{s}|$")
plt.show()
```

Read the four panels from left to right.

1. **Lab frame.** All the fluid nearby moves down with the sphere, at full speed on its surface. The
   streamlines bow away from the sphere in front of it (below) and close back in behind it. The colour
   fades slowly: that is the $1/r$ decay at work, and the disturbance is still visible several diameters
   away.
2. **Sphere frame.** The same field minus $\mathbf s$: a stream that comes from below, parts around the sphere
   and closes behind it. The picture is symmetric between front and back. Stokes flow has no wake.
3. **Stokeslet.** Almost the whole far field, but on its own it does not match the sphere velocity on the
   surface: it is brightest (speed $\tfrac32|\mathbf s|$, the top of the colour scale) at the front and
   back, and dimmer at the sides.
4. **Dipole.** A short-range correction. On the axis of motion it points *against* $\mathbf s$, at the sides
   *along* it. On the surface the Stokeslet alone gives $\tfrac32\mathbf s$ at the front and back and
   $\tfrac34\mathbf s$ at the sides; the dipole adds $-\tfrac12\mathbf s$ and $+\tfrac14\mathbf s$, so the sum is
   exactly $\mathbf s$ everywhere.

The figure uses the *unscreened* field so that you see the textbook solution. The toy's screened field has
the same near-field shape and dies off faster; Section 1.3 compares the two.

The animation below builds the same picture and then adds more spheres, which is where this lesson goes next.

<div class="course-anim" style="margin:1.2em 0 0.4em">
<video controls loop muted playsinline preload="metadata" poster="../_static/media/S3_superposition/S3_superposition.png" aria-label="Six spheres with glowing disturbance fields; at a marked point the velocity is built as the sum of one arrow per sphere; then the error of the naive sum lights up at a close pair." style="display:block;width:100%;border-radius:14px;background:#0f1419;box-shadow:0 4px 18px rgba(0,0,0,.25)">
<source src="../_static/media/S3_superposition/S3_superposition.webm" type="video/webm">
<source src="../_static/media/S3_superposition/S3_superposition.mp4" type="video/mp4">
Your browser cannot play this video. <a href="../_static/media/S3_superposition/S3_superposition.mp4">Download the MP4</a>.
</video>
</div>

**Animation S3, superposition.** One sphere first: in its own frame a uniform stream wraps around it, and a Galilean shift turns that into the lab-frame disturbance $G(\mathbf x - \mathbf r_1)\,\mathbf v_1$. Five more spheres appear, and at a marked point their contributions add tip to tail; distant spheres contribute almost nothing. The last part previews Sections 2 to 4 and Part III: each sphere also sits in the flow of the others (their surface average $W_i$), so its slip is $\mathbf s_i = \mathbf v_i - W_i$, not $\mathbf v_i$, and the error of the naive sum $\sum_j G\,\mathbf v_j$ is largest at the close pair. The clip is drawn on a dark stage in both page themes. Explore the same toy in 3D: <a href="../_static/media/S3_superposition/interactive.html" target="_blank" rel="noopener">open the interactive version</a> in a new tab, or show it here:

<details class="course-anim-interactive" style="margin:0.2em 0 1.4em">
<summary>Interactive 3D version: superposition versus self-consistent slip (toy)</summary>
<iframe src="../_static/media/S3_superposition/interactive.html" loading="lazy" title="superposition versus self-consistent slip (toy)" style="display:block;width:100%;height:580px;border:0;border-radius:14px;background:#0f1419;margin-top:0.6em"></iframe>
</details>

### 1.3 Checking the physics, and what screening costs

Every claim in Section 1.1 can be checked to round-off. No-slip is pointwise and exact for one sphere, with
or without screening.

```{code-cell} ipython3
dirs, _ = td.lebedev_sphere(31)                                # 350 directions on the unit sphere
surf = a * dirs
for L in (None, Lsc):
    u_s = td.stokes_sphere_velocity(surf, s_np, a, L)
    print(report(f"no-slip u = s on every surface point (L = {L})", np.abs(u_s - s_np).max(), 1e-12,
                 f"{len(surf)} points"))
```

For the Stokes equations we need derivatives. Autograd differentiates `stokes_u` analytically, so the
divergence and the Laplacian it returns are exact up to float64 round-off. We normalise each residual by
the size of the terms in it, because second derivatives grow like $1/r^3$ near the surface.

```{code-cell} ipython3
def derivatives(L, x):
    """Divergence, Laplacian and the field itself at points x (P, 3), by autograd in float64."""
    x = x.clone().requires_grad_(True)
    s = torch.as_tensor(s_np)
    u = stokes_u(x, s, a, L)
    grads = [torch.autograd.grad(u[:, i].sum(), x, create_graph=True)[0] for i in range(3)]   # du_i/dx_k
    div = sum(g[:, i] for i, g in enumerate(grads))
    lap = torch.stack([sum(torch.autograd.grad(g[:, k].sum(), x, retain_graph=True)[0][:, k] for k in range(3))
                       for g in grads], -1)
    return div.detach(), lap.detach(), grads

xq = torch.as_tensor(x_np[:300])
div, lap, grads = derivatives(None, xq)
grad_scale = torch.stack([g.detach().abs().amax(-1) for g in grads], -1).amax(-1)       # |du/dx| per point
print(report("unscreened: div u = 0", float((div.abs() / grad_scale).max()), 1e-12, "relative to |grad u|"))

xg = xq.clone().requires_grad_(True)
p = 1.5 * a * (xg * torch.as_tensor(s_np)).sum(-1) / xg.norm(dim=-1) ** 3        # mu = 1
grad_p = torch.autograd.grad(p.sum(), xg)[0]
mom = (lap - grad_p).abs().amax(-1) / (lap.abs().amax(-1) + grad_p.abs().amax(-1))
print(report("unscreened: mu lap u = grad p", float(mom.max()), 1e-12, "relative to the terms"))

div_s, lap_s, grads_s = derivatives(Lsc, xq)
gs_scale = torch.stack([g.detach().abs().amax(-1) for g in grads_s], -1).amax(-1)
rel_div = (div_s.abs() / gs_scale).numpy()
print(f"screened (L = {Lsc}): |div u| / |grad u| has median {np.median(rel_div):.2f}, max {rel_div.max():.2f}")
# momentum with the same pressure screened the same way, p S(r)
xg = xq.clone().requires_grad_(True)
p_s = 1.5 * a * (xg * torch.as_tensor(s_np)).sum(-1) / xg.norm(dim=-1) ** 3 * torch.exp(-(xg.norm(dim=-1) - a) / Lsc)
grad_ps = torch.autograd.grad(p_s.sum(), xg)[0]
mom_s = ((lap_s - grad_ps).abs().amax(-1) / (lap_s.abs().amax(-1) + grad_ps.abs().amax(-1))).numpy()
print(f"screened (L = {Lsc}): |lap u - grad p| relative to the terms has median {np.median(mom_s):.2f}")
```

The textbook field passes both tests at round-off. The screened field fails both: its divergence is a
sizeable fraction of its gradient, and momentum does not balance either (shown here with the pressure
screened by the same factor). So the toy's field is a *defined construction* that keeps the Stokes
shape near each sphere, not a solution of the Stokes equations. That is a deliberate **toy
simplification**: screening keeps the open-box sums finite and the neighbourhoods local, at the price of
mass and momentum conservation. A real suspension does not multiply the flow by a decaying factor. None of
the course's lessons rely on $\nabla\cdot\mathbf u = 0$.

How fast does the disturbance decay? Along the direction of motion and across it:

```{code-cell} ipython3
:tags: [hide-input]
r = np.geomspace(a, 30, 200)
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for k, (lab, d) in enumerate([("along $\\mathbf{s}$", np.array([0, 1.0, 0])), ("across $\\mathbf{s}$", np.array([1.0, 0, 0]))]):
    for L, ls in ((None, "-"), (Lsc, "--")):
        u = td.stokes_sphere_velocity(r[:, None] * d, np.array([0, -1.0, 0]), a, L)
        ax.loglog(r / (2 * a), np.linalg.norm(u, axis=1), ls, color=cp.PALETTE[k],
                  label=f"{lab}, {'unscreened' if L is None else f'screened L = {L}'}")
ax.loglog(r / (2 * a), 1.5 * a / r, ":", color=cp.INK, lw=1.2, label=r"$3a/(2r)$ (Stokeslet, along $\mathbf{s}$)")
ax.set_xlabel("distance from the centre $r/d$"), ax.set_ylabel(r"$|\mathbf{u}| / |\mathbf{s}|$")
ax.set_ylim(1e-6, 1.5), ax.set_title("one-sphere disturbance: slow $1/r$ decay vs screening")
ax.legend(fontsize=8)
plt.show()
```

Unscreened, the disturbance falls only like $1/r$: along the direction of motion it is still about
$3a/(2r)$, which is 3 % of $|\mathbf s|$ at 25 diameters. Across the motion it is half that. The dipole
matters only within a diameter or two, where the curves bend away from the dotted line. The screened
curves drop exponentially after a couple of diameters.

The $1/r$ decay is why one sphere's influence reaches so far, and why summing many unscreened Stokeslets in a
large box does not converge: there are about $4\pi r^2\,dr\cdot n$ spheres at distance $r$, each contributing
$\sim 1/r$, so the sum grows with the box. In a real sedimenting suspension the *mean* is fixed by the
backflow and the mean pressure gradient (Batchelor's renormalisation); whether the velocity *fluctuations*
are screened is a long-debated question (Caflisch and Luke), and Brinkman-type screening is firmly
established only for fixed beds of particles {cite}`guazzelli2011physical`. The toy sidesteps all of this
with $e^{-(r-a)/L}$.

### 1.4 Symmetry: why the kernel lies in a plane

Expand $G(\mathbf x)\mathbf s$ and collect terms. Every term is either along $\mathbf x$ or along
$\mathbf s$:

$$
G(\mathbf x)\,\mathbf s = \alpha(r, \mathbf x\cdot\mathbf s)\,\mathbf x + \beta(r)\,\mathbf s,\qquad
\alpha = \Bigl(\frac{3a}{4r^3} - \frac{3a^3}{4r^5}\Bigr)(\mathbf x\cdot\mathbf s)\,S(r),\qquad
\beta = \Bigl(\frac{3a}{4r} + \frac{a^3}{4r^3}\Bigr)S(r),
$$

with $S(r) = e^{-(r-a)/L}$ (or 1). The coefficients depend only on the scalars $r$ and
$\mathbf x\cdot\mathbf s$, which no rotation or reflection changes. Two consequences follow, and both are
checked below.

- The velocity lies in $\mathrm{span}(\mathbf x, \mathbf s)$. This is exactly the form of every message of
  the A0 network ([Lesson 02c](../part2_a0/02c_span_property.md)), which replaces $\alpha$ and $\beta$ by
  learned functions of more scalars. Those include $|\mathbf q_j|$, so a learned message need not be linear
  in the state, whereas $G\mathbf s$ is.
- The map $(\mathbf x, \mathbf s) \mapsto G(\mathbf x)\mathbf s$ is $O(3)$-equivariant: rotate or reflect both
  inputs and the output rotates or reflects with them.

Geometrically: the offset $\mathbf x$ and the slip $\mathbf s$ span a plane through the query point, and
each sphere's contribution is an arrow in its own plane. With several spheres the planes differ, so the
*sum* points in a direction none of them contains. The animation still shows this for three spheres, in
the notation of the A0 network (message $\mathbf m_j$, total $\mathbf u_f$).

```{figure} ../_static/media/S2_span_plane/S2_span_plane.png
:width: 100%
:alt: Three spheres with velocity arrows; from each sphere a translucent plane passes through a query point f, each containing one message arrow; the summed arrow u_f lies in none of the planes.

Every one-particle message lies in the plane spanned by the offset to the query point and the particle's
state; the sum $\mathbf u_f = \sum_j \mathbf m_j$ is a general 3D vector. The full animation S2, with an
interactive 3D version, plays in [Lesson 02c, Section 1](../part2_a0/02c_span_property.md).
```

```{code-cell} ipython3
def alpha_beta_form(x, s, L=Lsc):
    r = x.norm(dim=-1, keepdim=True)
    S = torch.exp(-(r - a) / L)
    xs = (x * s).sum(-1, keepdim=True)
    return (3 * a / (4 * r**3) - 3 * a**3 / (4 * r**5)) * xs * S * x + (3 * a / (4 * r) + a**3 / (4 * r**3)) * S * s

xt = torch.as_tensor(x_np)
st = torch.as_tensor(rng.normal(size=x_np.shape))               # a different slip at every point
u_t = stokes_u(xt, st, a, Lsc)
print(report("G s == alpha x + beta s", float((u_t - alpha_beta_form(xt, st)).abs().max()), 1e-12))
print(report("G s lies in span(x, s)", float(span_residual(u_t, xt, st).max()), 1e-12, "relative residual"))
check_equivariance(lambda x, s: stokes_u(x, s, a, Lsc), (xt, st), ("vector", "vector"),
                   name="one-sphere kernel (x, s) -> G(x) s");
```

What breaks it? Suppose someone "simplifies" the kernel by always pointing the Stokeslet's extra term
down, along gravity, instead of along $\hat{\mathbf x}$, the direction from the sphere to the query point.
The result looks plausible for
settling spheres, but it uses a fixed direction that does not rotate with the inputs.

```{code-cell} ipython3
down = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)

def kernel_with_fixed_direction(x, s):
    r = x.norm(dim=-1, keepdim=True)
    return 3 * a / (4 * r) * (s + down * (down * s).sum(-1, keepdim=True))   # x^ replaced by gravity

try:
    check_equivariance(kernel_with_fixed_direction, (xt, st), ("vector", "vector"), name="kernel with gravity built in")
except AssertionError as err:
    print("caught:", err)
```

A third property is special to Stokes flow. Because $\alpha\,\mathbf x$ is even in $\mathbf x$ (both
factors flip) and $\beta$ depends only on $r$, the field is **fore-aft symmetric**:
$\mathbf u(-\mathbf x) = \mathbf u(\mathbf x)$. The flow in front of a sphere mirrors the flow behind it.

```{code-cell} ipython3
fa = (stokes_u(-xt, st, a, Lsc) - u_t).abs().max()
print(report("fore-aft symmetry u(-x) = u(x)", float(fa), 1e-12))
```

At the particle Reynolds numbers of the research simulations this symmetry is gone: inertia leaves a wake
behind each sphere. A model built from $\mathbf x$, $\mathbf s$ and invariants can represent a wake, because
nothing forces $\alpha$ to be odd in $\mathbf x\cdot\mathbf s$, but it has to *learn* the shape from data. This
is the first reason the surrogate learns its one-particle kernel instead of using $G$.

## 2. Adding disturbances together

### 2.1 Linearity

The Stokes equations are linear: if $(\mathbf u_1, p_1)$ and $(\mathbf u_2, p_2)$ are solutions, so is their
sum. So it is tempting to build the flow of $N$ spheres as

$$
\mathbf u(\mathbf x) = \mathbf U_b + \sum_{j=1}^N G(\mathbf x - \mathbf r_j)\,\mathbf s_j .
$$

The toy's field is exactly this sum, with $\mathbf s_j$ the self-consistent slips (the cell below only
confirms that the toy is built this way; it is a consistency check, not a test of physics). The open
question is which velocity $\mathbf s_j$ each sphere should carry. Linearity says nothing about it.

```{code-cell} ipython3
snap = make_snapshot(cfg, seed=0)                       # 96 spheres, the course default
pts = snap.sample_exterior(2000, rng=np.random.default_rng(2))
G_all = stokes_sphere_matrix(pts[:, None, :] - snap.positions[None], a, Lsc)     # (P, N, 3, 3)
superpose = lambda states: snap.backflow + np.einsum("pnij,nj->pi", G_all, states)
true = snap.field(pts)
print(report("consistency (same construction): toy field == U_b + sum_j G(x - r_j) s_j",
             np.abs(superpose(snap.slip) - true).max(), 1e-12))
```

### 2.2 Two spheres: the obvious guess fails

Take two spheres, both moving down with velocity $\mathbf v$, with centres a distance $D$ apart. The obvious
guess is to let each sphere radiate its own velocity, $\mathbf s_j = \mathbf v_j$. Then the field on sphere 1's
surface is $\mathbf v_1 + G(\mathbf x - \mathbf r_2)\mathbf v_2$, not $\mathbf v_1$. The second sphere's flow
comes on top, and no-slip is broken. Sphere 1 sits in the fluid that sphere 2 drags down, so it moves *more
slowly relative to that fluid* than relative to the fluid far away.

The toy's correction is the condition from Lesson 01a, Section 3.4: choose the slips so that the field
averaged over each sphere's surface equals that sphere's velocity,

$$
\mathbf s_i = \mathbf v_i - \mathbf U_b - \widetilde W_i(\mathbf s), \qquad
\widetilde W_i(\mathbf s) = \Bigl\langle \sum_{j \ne i} G(\cdot - \mathbf r_j)\,\mathbf s_j \Bigr\rangle_{\text{surface of } i},
$$

where $\widetilde W_i$ is the exact flow the *other* spheres make around sphere $i$, averaged with the
toy's 50-point Lebedev rule. (The tilde marks the toy's exact interaction; plain $W_i$ is reserved for the
model's prediction in Part III.) The slip $\mathbf s_i$ is the particle's velocity relative to the fluid
around it: the **state** $\mathbf q_i$ of [Lesson 01a](01a_settling_particles.md). It is linear in
$\mathbf s$, so for $N$ spheres it is a $3N \times 3N$ linear system, $(I + A)\,\mathbf s = \mathbf v - \mathbf U_b$.

We build two-sphere snapshots with a small builder, written from the toy's public pieces as in Lesson 01a.
For pairs we switch the backflow off. Naive superposition is then `self_consistent=False`, which sets
$\mathbf s = \mathbf v - \mathbf U_b$.

```{code-cell} ipython3
def build_snapshot(positions, velocity, cfg, grid=None):
    """The toy's truth for given particle arrays (same steps as make_snapshot)."""
    grid = np.eye(3) if grid is None else np.asarray(grid)
    phi = cfg.volume_fraction
    ub = -phi / (1 - phi) * velocity.mean(0) if cfg.backflow else np.zeros(3)
    slip = td.solve_slip(positions, velocity, ub, cfg, grid)
    return ToySnapshot(positions, velocity, slip, ub,
                       td.local_volume_fraction(positions, cfg.radius, cfg.phi_radius), cfg, grid)

def surface_average(sn):
    """Field averaged over each sphere's surface with the toy's 50-point Lebedev rule: (N, 3)."""
    sp, w = td.surface_points(sn.positions, sn.config.radius, sn.config.lebedev_order, grid=sn.grid)
    return np.einsum("l,nlc->nc", w, sn.field(sp.reshape(-1, 3)).reshape(sn.n, -1, 3))

cfg2 = replace(cfg, n_particles=2, backflow=False)
D = 1.4                                                   # centre distance, in units of d = 1
pair_pos = centre + np.array([[-D / 2, 0, 0], [D / 2, 0, 0]])          # side by side
pair_vel = np.array([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]])
pair_naive = build_snapshot(pair_pos, pair_vel, replace(cfg2, self_consistent=False))
pair_true = build_snapshot(pair_pos, pair_vel, cfg2)
print("naive slips:\n", pair_naive.slip.round(3), "\nself-consistent slips:\n", pair_true.slip.round(3))

for name, sn in (("naive superposition", pair_naive), ("self-consistent slips", pair_true)):
    err = np.abs(surface_average(sn) - sn.velocity).max()
    try:
        assert err < 1e-12, f"surface average misses the particle velocity by {err:.3f}"
        print(report(f"{name}: <u>_surface == v", err, 1e-12))
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
```

Naive superposition fails the averaged no-slip condition by about a tenth of the velocity at this
separation (printed above). The self-consistent slips pass. Both slips are smaller than $|\mathbf v|$: each sphere is partly carried by the
other's flow. (In real Stokes flow two spheres that are pulled by the same force fall *faster* together
than apart. The toy prescribes the velocities instead of the forces, so the same physics appears as a
smaller slip.)

The average is right, but what about individual surface points? The polar plot walks once around sphere 1's
equator in the $xy$ plane (the plane through both centres), starting at the side that faces sphere 2.

```{code-cell} ipython3
:tags: [hide-input]
theta = np.linspace(0, 2 * np.pi, 361)
ring = pair_pos[0] + a * np.stack([np.cos(theta), np.sin(theta), 0 * theta], -1)   # theta = 0 faces sphere 2
for lab, sn in (("naive", pair_naive), ("self-consistent", pair_true)):
    dev = np.linalg.norm(sn.field(ring) - sn.velocity[0], axis=1)
    print(f"{lab:16s} |u - v_1| on the ring: {dev[0]:.3f} facing sphere 2, {dev[180]:.3f} on the far side")
fig = plt.figure(figsize=(11.5, 4.4))
ax0 = fig.add_subplot(1, 2, 1)
cp.field_slice(pair_true, ax=ax0, n=121, streamlines=True, vmax=1.0, title=f"self-consistent pair, $D = {D}\\,d$")
ax0.set_xlim(2, 6), ax0.set_ylim(2, 6)
ax0.plot(ring[:, 0], ring[:, 1], color=cp.PALETTE[4], lw=3.0, zorder=4)
ax1 = fig.add_subplot(1, 2, 2, projection="polar")
for k, (lab, sn) in enumerate((("naive (s = v)", pair_naive), ("self-consistent s", pair_true))):
    dev = np.linalg.norm(sn.field(ring) - sn.velocity[0], axis=1)
    ax1.plot(theta, dev, color=cp.PALETTE[k], label=lab)
ax1.set_title(r"$|\mathbf{u} - \mathbf{v}_1|$ around sphere 1 (0 = facing sphere 2)", fontsize=10)
ax1.tick_params(colors=cp.INK), ax1.grid(color=cp.GRID), ax1.set_rlabel_position(115)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
plt.tight_layout()
plt.show()
```

On the left, the two spheres drag a shared column of fluid down; the magenta ring marks the path of the polar
plot. On the right, the naive curve (blue) is largest on the side that faces sphere 2, where sphere 2's
flow is strongest, and small on the far side. The self-consistent slip (orange) removes the *average* of
that extra flow: the mismatch facing the neighbour drops to about half, and on the far side it is about
as large as before, or slightly larger, because one uniform slip correction cannot follow a flow that
varies across the sphere (the printed values). The surface *average* is exact, the pointwise values are
not.

Removing that pointwise remainder would need the near-field corrections of the full **method of
reflections** {cite}`happel1983low`: higher multipoles on each sphere that cancel the neighbour's flow point
by point, then the flow those make, and so on. The toy keeps only the uniform part of each correction (one
slip vector per sphere). Sections 2.3 and 4.1 apply the same reflect-and-correct idea to that reduced,
surface-averaged system; we call it **reflections of the averaged system** to keep the two apart. Real simulations contain all of
it, plus lubrication between nearly touching spheres. This is the second reason the surrogate learns its
kernel: a learned pair message can absorb near-field effects that a single-sphere formula with one slip
vector cannot.

### 2.3 How the error depends on separation

Now we sweep the separation $D$ for two orientations of the pair: **side by side** (offset across the
motion) and **in line** (one behind the other). For each we record the relative slip correction
$|\mathbf s_1 - \mathbf v_1| / |\mathbf v_1|$, which is exactly what naive superposition gets wrong, and
compare with the **first reflection** of the averaged system, $\mathbf s \approx \mathbf v - A\mathbf v$:
the neighbour's flow evaluated with the naive slips.

```{code-cell} ipython3
Ds = np.linspace(1.06, 6.0, 60)
sweep = {}
for orient, axis in (("side by side", 0), ("in line", 1)):
    for L in (Lsc, None):
        c2 = replace(cfg2, screening_length=L)
        exact, first = [], []
        for Dk in Ds:
            off = np.zeros(3); off[axis] = Dk / 2
            pos = centre + np.stack([-off, off])
            A = td.interaction_matrix(pos, c2)
            b = pair_vel.reshape(-1)
            sc = np.linalg.solve(np.eye(6) + A, b).reshape(2, 3)
            exact.append(np.linalg.norm(sc[0] - pair_vel[0]))
            first.append(np.linalg.norm((A @ b).reshape(2, 3)[0]))
        sweep[orient, L] = (np.array(exact), np.array(first))

# consistency (same construction): the builder calls the same solve, at the separation used above
print(report("consistency: direct 6x6 solve == build_snapshot slip (side by side, D = 1.4)",
             np.abs(np.linalg.solve(np.eye(6) + td.interaction_matrix(pair_pos, cfg2), pair_vel.reshape(-1))
                    .reshape(2, 3) - pair_true.slip).max(), 1e-12))
```

```{code-cell} ipython3
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, L in zip(axes, (Lsc, None)):
    for k, orient in enumerate(("side by side", "in line")):
        exact, first = sweep[orient, L]
        ax.plot(Ds, exact, color=cp.PALETTE[k], label=f"{orient}: self-consistent")
        ax.plot(Ds, first, "--", color=cp.PALETTE[k], label=f"{orient}: first reflection")
    ax.set_xlabel("centre distance $D/d$"), ax.set_title("screened, $L = 1$" if L else "unscreened (textbook $G$)")
    ax.set_yscale("log"), ax.legend(fontsize=8)
axes[0].set_ylabel(r"$|\mathbf{s}_1 - \mathbf{v}_1| / |\mathbf{v}_1|$  (naive superposition error)")
plt.tight_layout()
plt.show()
for orient in ("side by side", "in line"):
    e = sweep[orient, Lsc][0]
    print(f"screened, {orient:12s}: slip correction {e[0]:.2f} at contact, {e[np.argmin(abs(Ds - 2))]:.3f} at D = 2d, "
          f"{e[np.argmin(abs(Ds - 4))]:.4f} at D = 4d")
```

Three things to read off:

- **Near contact the correction is large**, a sizeable fraction of the velocity, and it falls quickly with
  distance. In the screened toy (left) it falls exponentially; with the textbook kernel (right) only like
  $1/D$, so even distant pairs matter.
- **A sphere in line is corrected more than one side by side.** The Stokeslet along the direction of motion is twice as strong as across it
  (Section 1.3), so a sphere directly behind another feels more of its flow.
- **The first reflection is good when the spheres are far apart and overshoots near contact.** Using the
  naive slips to compute the neighbour's flow counts the neighbour as moving faster relative to the fluid
  than it really does. Each further reflection corrects the previous one. With two spheres the series
  converges. Section 4 shows that with many spheres it need not.

## 3. Many spheres: where naive superposition goes wrong

### 3.1 The failure on a full snapshot

Back to the default snapshot with 96 spheres. Naive superposition lets each sphere radiate its **base
state**: its velocity in the toy's reference frame,

$$
\mathbf q^{\text{base}}_j = \mathbf v_j - \mathbf U_b ,
$$

the same definition Lesson 03a uses. It is the right-hand side of the toy's equation
$(I + A)\,\mathbf s = \mathbf v - \mathbf U_b$, so it is what the slips would be if the spheres did not
interact. (The research used the *measured* exterior mean $\langle\mathbf u_f\rangle$ as its frame instead,
see [Lesson 01c](01c_frames.md); Section 3.2 shows that this choice matters.) The toy provides this baseline
as `make_snapshot(..., self_consistent=False)`, with identical positions and velocities. First we confirm
that it is the inline sum (a consistency check), then we test the averaged no-slip condition.

```{code-cell} ipython3
naive = make_snapshot(cfg, seed=0, self_consistent=False)
q_base = snap.velocity - snap.backflow
print(report("consistency: same particles in both snapshots", np.abs(naive.positions - snap.positions).max()
             + np.abs(naive.velocity - snap.velocity).max(), 1e-12))
u_naive = superpose(q_base)
print(report("consistency: naive snapshot field == U_b + sum_j G(x - r_j) q_base_j",
             np.abs(naive.field(pts) - u_naive).max(), 1e-12))

for name, sn in (("naive superposition", naive), ("self-consistent", snap)):
    err = np.abs(surface_average(sn) - sn.velocity).max() / np.abs(sn.velocity).max()
    try:
        assert err < 1e-12, f"worst surface average misses v by {err:.2f} (relative to max |v|)"
        print(report(f"{name}: <u>_surface == v", err, 1e-12, f"{sn.n} spheres"))
    except AssertionError as e:
        print(f"FAIL  {name}: {e}")
```

For the worst sphere, the naive surface average misses the particle velocity by an order-one fraction of
the largest particle speed (the printed number is a maximum over spheres and components). How much does
that matter for the *field*? We measure the error relative to the RMS of the true fluctuation
$\mathbf u - \overline{\mathbf u}$ (the bar is the mean over the sample points). We report it twice: raw,
and with the mean error removed, because a uniform offset of the output is absorbed by the reference frame
([Lesson 01c](01c_frames.md)).

```{code-cell} ipython3
def field_errors(u_model, u_true):
    fl = u_true - u_true.mean(0)
    rms = np.sqrt((fl**2).sum(1).mean())
    d = u_model - u_true
    return np.sqrt((d**2).sum(1).mean()) / rms, np.sqrt(((d - d.mean(0))**2).sum(1).mean()) / rms

raw, centred = field_errors(u_naive, true)
print(f"naive superposition, 96 spheres: relative field error {raw:.2f} raw, {centred:.2f} with the mean removed")
print(f"mean error (a uniform offset): {np.round((u_naive - true).mean(0), 3)}")
```

### 3.2 Most of the miss is a collective drift

Before looking for local structure, look at what the self-consistent solve changes: the correction
$\mathbf s_j - \mathbf q^{\text{base}}_j$ for every sphere. The figure draws both arrows on the spheres
that the mid-plane slice cuts: the base state (what naive superposition radiates) and the slip (what the
truth radiates).

```{code-cell} ipython3
:tags: [hide-input]
near = np.abs(snap.positions[:, 2] - cfg.box / 2) < a
fig, ax = plt.subplots(figsize=(6.2, 6.2))
for c in snap.positions[near]:
    dz = abs(c[2] - cfg.box / 2)
    ax.add_patch(plt.Circle(c[:2], np.sqrt(a * a - dz * dz), facecolor=cp.GRID, edgecolor=cp.INK, lw=0.8))
P = snap.positions[near]
for k, (vec, lab) in enumerate(((q_base, r"base state $\mathbf{q}^{\mathrm{base}}_j = \mathbf{v}_j - \mathbf{U}_b$"),
                               (snap.slip, r"slip $\mathbf{s}_j$ (the true state)"))):
    ax.quiver(P[:, 0], P[:, 1], vec[near, 0], vec[near, 1], color=cp.PALETTE[k], angles="xy", scale_units="xy",
              scale=1.0, width=0.006, label=lab, zorder=3 + k)
ax.set_xlim(0, cfg.box), ax.set_ylim(-1, cfg.box), ax.set_aspect("equal")
ax.set_xlabel("x"), ax.set_ylabel("y"), ax.legend(loc="lower right", fontsize=8)
ax.set_title("state = slip: spheres radiate less than their own velocity")
plt.show()
```

Almost every slip is shorter than its base state, and most are shortened along the same downward
direction: the whole cloud of spheres drags the fluid inside it down, so a sphere moves more slowly
relative to its surroundings than relative to the toy's frame. A few slips also turn, where a close
neighbour's flow has a sideways component. Split the correction into its mean over the particles and the rest, and look at which
eigenvectors of $A$ carry the largest eigenvalues.

```{code-cell} ipython3
A96 = td.interaction_matrix(snap.positions, cfg, snap.grid)
corr = snap.slip - q_base
c_mean = corr.mean(0)
print(f"correction s - q_base: mean over particles {np.round(c_mean, 3)} (|mean| = {np.linalg.norm(c_mean):.3f}),"
      f" RMS of the rest {np.sqrt(((corr - c_mean)**2).sum(1).mean()):.3f}")

lam, V = np.linalg.eigh(A96)                       # A is symmetric (Lebedev averages of a symmetric kernel)
U3 = np.kron(np.ones((snap.n, 1)), np.eye(3)) / np.sqrt(snap.n)   # orthonormal basis: translate all spheres
for k in (1, 2, 3, 4):
    print(f"eigenvalue {lam[-k]:.3f}: overlap with uniform translation {np.linalg.norm(U3.T @ V[:, -k])**2:.2f}")
Pu = np.eye(3 * snap.n) - U3 @ U3.T
rho_full = np.abs(lam).max()
rho_proj = np.abs(np.linalg.eigvalsh(Pu @ A96 @ Pu)).max()
print(f"spectral radius of A: {rho_full:.2f}; with the 3 uniform modes projected out: {rho_proj:.2f}")
```

The mean correction is several times larger than the particle-to-particle variation, and the three
largest eigenvalues of $A$ belong mostly to uniform translation of all spheres. This is the **collective
drift** of Lesson 01a, Section 2.2: in the toy's open box the spheres drag a whole column of fluid with
them, and the self-consistent slips subtract it. Naive superposition does not, so its miss of the averaged
no-slip condition, and much of its field error, is this one collective mode rather than local crowding.
It is a property of the toy's open box, and it is also why the choice of frame for the *states* matters.
Compare the toy's frame with the research frame, the measured exterior mean, by superposing
$\mathbf c + \sum_j G(\mathbf x - \mathbf r_j)(\mathbf v_j - \mathbf c)$ for both choices of $\mathbf c$.

```{code-cell} ipython3
u_f = snap.exterior_mean()
print(f"U_b = {np.round(snap.backflow, 3)},  measured <u_f> = {np.round(u_f, 3)}")
print("base state        raw err  centred err  slip error ||q - s|| / ||s||")
for lab, c in (("v - U_b", snap.backflow), ("v - <u_f>", u_f)):
    q = snap.velocity - c
    r_, c_ = field_errors(c + np.einsum("pnij,nj->pi", G_all, q), true)
    print(f"{lab:16s}  {r_:6.2f}   {c_:8.2f}    {np.linalg.norm(q - snap.slip) / np.linalg.norm(snap.slip):8.2f}")
```

The measured mean already contains most of the collective drift, so subtracting it from the velocities
removes much of the drift from the states as well, and every error drops (by the printed factors). This
is *not* a uniform shift of the output: changing the states changes every radiated $G\,\mathbf q_j$
(Exercise 3 shows that a uniform output shift alone cannot do this). But neither frame satisfies averaged
no-slip, and what remains is the local part. For the rest of the lesson we stay with
$\mathbf v - \mathbf U_b$, the frame in which the toy's equation is written.

### 3.3 Where the error lives

The slice below shows the true fluctuation, the naive one, and the difference, in the frame of each
field's own mean so that the uniform offset does not dominate. The difference panel has its own colour
scale.

```{code-cell} ipython3
:tags: [hide-input]
n_sl = 110
sl_pts, sl_mask, _, _ = td.slice_grid(snap, "z", None, n_sl)
u_true_sl = np.full((n_sl * n_sl, 3), np.nan); u_true_sl[sl_mask] = snap.field(sl_pts[sl_mask])
u_naive_sl = np.full((n_sl * n_sl, 3), np.nan); u_naive_sl[sl_mask] = naive.field(sl_pts[sl_mask])
mt, mn = np.nanmean(u_true_sl, 0), np.nanmean(u_naive_sl, 0)
err_sl = (u_naive_sl - mn) - (u_true_sl - mt)
vmax = np.nanpercentile(np.hypot(*(u_true_sl - mt)[:, :2].T), 99)
vmax_err = np.nanpercentile(np.hypot(*err_sl[:, :2].T), 99)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
cp.field_slice(snap, values=u_true_sl - mt, ax=axes[0], n=n_sl, streamlines=True, vmax=vmax,
               title="truth (self-consistent slips)")
cp.field_slice(snap, values=u_naive_sl - mn, ax=axes[1], n=n_sl, streamlines=True, vmax=vmax,
               title="naive superposition ($\\mathbf{s} = \\mathbf{v} - \\mathbf{U}_b$)")
cp.field_slice(snap, values=err_sl, ax=axes[2], n=n_sl, vmax=vmax_err, colorbar=False,
               title="difference (mean removed): in-plane magnitude")
fig.colorbar(axes[2].images[0], ax=axes[2], shrink=0.85, label="in-plane error")
for ax in axes:
    ax.set_xlim(1, 7), ax.set_ylim(1, 7)
plt.tight_layout()
plt.show()
```

Compare the left and middle panels. Naive superposition gets the *pattern* roughly right, since every sphere
still drags fluid in the right direction, but it exaggerates the disturbances: each sphere radiates its full
velocity even though it is moving inside a stream that already carries it partly along. Even with the mean
removed, the error (right) is not uniform: it is largest around some of the spheres and smaller in open
fluid. That is consistent with the left panel of Section 3.4: crowded spheres carry more than the average
share of the drift, so removing the mean leaves error around them. That structure matters for a
learned model: the correction depends on *where the neighbours are*, not only on the particle itself.

### 3.4 How the error grows with crowding

Per particle, plot the component of the correction along the mean correction direction (the drift
direction) against each particle's local volume fraction $\phi_j$ (the neighbour count of Lesson 01a).
Then sweep the global volume fraction by changing the number of spheres in the same box, two seeds per
value. The table prints the numerator and the denominator of the centred error separately.

```{code-cell} ipython3
Ns = [16, 32, 48, 72, 96, 120, 144]
rows = []
for N in Ns:
    for seed in (0, 1):
        sn = make_snapshot(cfg, n_particles=N, seed=seed)
        p_ = sn.sample_exterior(1200, rng=np.random.default_rng(100 + seed))
        G_ = stokes_sphere_matrix(p_[:, None, :] - sn.positions[None], a, Lsc)
        u_t = sn.field(p_)
        u_n = sn.backflow + np.einsum("pnij,nj->pi", G_, sn.velocity - sn.backflow)
        d_ = u_n - u_t
        A = td.interaction_matrix(sn.positions, sn.config, sn.grid)
        rows.append((N, sn.config.volume_fraction, *field_errors(u_n, u_t),
                     np.abs(np.linalg.eigvalsh(A)).max(),
                     np.sqrt(((d_ - d_.mean(0))**2).sum(1).mean()),            # centred error, absolute
                     np.sqrt(((u_t - u_t.mean(0))**2).sum(1).mean())))         # RMS of the true fluctuation
rows = np.array(rows)
print(" N    phi    raw err  centred err  spectral radius | centred err (abs)  true fluct RMS")
for r_ in rows[::2]:
    print(f"{int(r_[0]):3d}  {r_[1]:.3f}   {r_[2]:6.2f}   {r_[3]:6.2f}      {r_[4]:6.2f}      |"
          f"   {r_[5]:6.3f}          {r_[6]:6.3f}")
```

```{code-cell} ipython3
:tags: [hide-input]
drift_dir = c_mean / np.linalg.norm(c_mean)
along = corr @ drift_dir / np.linalg.norm(q_base, axis=1)
print(f"correlation of the along-drift correction with phi_j: {np.corrcoef(snap.phi_local, along)[0, 1]:.2f}")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
axes[0].scatter(snap.phi_local, along, s=14, color=cp.PALETTE[2], alpha=0.75, edgecolors="none")
axes[0].axhline(along.mean(), color=cp.INK, lw=1, ls=":", label="mean (the collective drift)")
axes[0].set_xlabel(r"local volume fraction $\phi_j$"), axes[0].legend(fontsize=8)
axes[0].set_ylabel(r"$(\mathbf{s}_j - \mathbf{q}^{\mathrm{base}}_j)\cdot\hat{\mathbf{e}}_{\mathrm{drift}}"
                   r" / |\mathbf{q}^{\mathrm{base}}_j|$")
axes[0].set_title("per particle: a shared drift, larger where crowded")
phis = rows[:, 1]
axes[1].plot(phis, rows[:, 3], "o", color=cp.PALETTE[0], label="field error, mean removed")
axes[1].plot(phis, rows[:, 2], "s", mfc="none", color=cp.PALETTE[1], label="field error, raw")
for col, c in ((3, cp.PALETTE[0]), (2, cp.PALETTE[1])):
    m = [rows[rows[:, 0] == N, col].mean() for N in Ns]
    axes[1].plot(sorted(set(phis)), m, color=c, lw=1.2)
axes[1].set_xlabel(r"global volume fraction $\phi$"), axes[1].set_ylabel("naive error / RMS of true fluctuation")
axes[1].set_title("naive superposition gets worse with crowding"), axes[1].legend()
plt.tight_layout()
plt.show()
```

Left: every sphere's correction has a shared part (the dotted line), which is the collective drift of
Section 3.2; even isolated spheres get a sizeable fraction of it, so the floor of the scatter is the drift,
not a neighbour. On top of it, crowded spheres are corrected more (the printed correlation), but the scatter
is wide: the local count does not fix the correction, because the *arrangement* of the neighbours matters
too.

Right: naive superposition is a fair approximation only for very dilute suspensions. Its field error grows
with the volume fraction, and from the course's default ($\phi \approx 0.1$) upwards it is about as large
as the fluctuation itself, even after removing the uniform offset. The two right-hand table columns show
why the centred curve then flattens: the true fluctuation RMS barely changes with $\phi$, and the absolute
centred error stops growing at the largest $\phi$ values. With only two seeds per point, that flattening
is within the seed-to-seed scatter, so read it as "no longer growing quickly", not as a limit. The raw
error keeps growing, because the collective drift grows with the number of spheres.

## 4. The fix: a state found self-consistently

### 4.1 Reflections of the averaged system

The self-consistent slips solve $(I + A)\,\mathbf s = \mathbf b$ with $\mathbf b = \mathbf q^{\text{base}}
= \mathbf v - \mathbf U_b$. A physicist's first instinct is to reflect: start from the naive slips,
subtract the flow the others make, then correct for the correction, and so on,

$$
\mathbf s^{(K)} = \sum_{k=0}^{K} (-A)^k\,\mathbf b
= \mathbf b - A\mathbf b + A^2\mathbf b - \dots
$$

$K = 0$ is naive superposition and $K = 1$ is the first reflection of Section 2.3. This geometric series
converges exactly when every eigenvalue of $A$ has magnitude below 1, that is, when the *spectral radius*
$\varrho(A) < 1$. The table above printed $\varrho(A)$; it passes 1 somewhere between the dilute and the default
volume fractions. Here is what the series does on either side.

```{code-cell} ipython3
def reflections(sn, K=12):
    A = td.interaction_matrix(sn.positions, sn.config, sn.grid)
    b = (sn.velocity - sn.backflow).reshape(-1)
    s_k, term, errs = b.copy(), b.copy(), []
    errs.append(np.linalg.norm(s_k - sn.slip.reshape(-1)) / np.linalg.norm(sn.slip))
    for _ in range(K):
        term = -A @ term
        s_k = s_k + term
        errs.append(np.linalg.norm(s_k - sn.slip.reshape(-1)) / np.linalg.norm(sn.slip))
    return np.array(errs), np.abs(np.linalg.eigvals(A)).max()

refl = {N: reflections(make_snapshot(cfg, n_particles=N, seed=0)) for N in (16, 48, 96)}
for N, (e, rho) in refl.items():
    print(f"N = {N:3d}: spectral radius {rho:.2f}; slip error after 0, 1, 4, 12 reflections: "
          f"{e[0]:.2e}, {e[1]:.2e}, {e[4]:.2e}, {e[12]:.2e}")
```

```{code-cell} ipython3
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for k, (N, (e, rho)) in enumerate(refl.items()):
    phi = make_snapshot(cfg, n_particles=N, seed=0).config.volume_fraction
    ax.semilogy(np.arange(len(e)), e, "o-", ms=4, color=cp.PALETTE[k],
                label=rf"$\phi = {phi:.2f}$, $\varrho(A) = {rho:.2f}$")
ax.axhline(1.0, color=cp.INK, lw=1, ls=":")
ax.set_xlabel("number of reflections $K$"), ax.set_ylabel(r"$\|\mathbf{s}^{(K)} - \mathbf{s}\| / \|\mathbf{s}\|$")
ax.set_title("reflections of the averaged system converge only when dilute"), ax.legend()
plt.show()
```

In the dilute cases each reflection shrinks the error by a constant factor. At the default volume fraction
the series **diverges**: each correction overshoots by more than the last. The largest eigenvalues belong
to the collective drift of Section 3.2, but the divergence is not only a box artifact: with the three
uniform modes projected out, the spectral radius printed there is still above 1. The series is the same
thing as repeatedly substituting $\mathbf s \leftarrow \mathbf b - A\mathbf s$ starting from
$\mathbf s = \mathbf b$, which Part III calls undamped Picard iteration, so its divergence there is this
same fact. Solving the linear system directly is one fix, and the toy does exactly that. **Damping** the
iteration is another, $\mathbf s \leftarrow (1 - \alpha)\,\mathbf s + \alpha(\mathbf b - A\mathbf s)$:
each eigenvalue $\mu$ of $I + A$ is then multiplied by $1 - \alpha\mu$ per step, and every factor has
magnitude below 1 when $0 < \alpha < 2/\mu_{\max}$. Part III uses damping, because there the map is not a
known matrix but a trained network.

```{figure} ../_static/media/S4_fixed_point/S4_fixed_point.png
:width: 100%
:alt: Two panels of spheres with state arrows; with step alpha 0.5 the arrows settle, with alpha 0.8 they grow; a residual plot and a number line show the top-mode factor 1 - alpha mu.

Damped iteration on a small stylised arrangement: a step $\alpha = 0.5$ converges, $\alpha = 0.8$ is too
large and the top mode grows, because $|1 - \alpha\mu_{\max}| > 1$. The values of $\mu_{\max}$ and the
bound on $\alpha$ in the still belong to the animation's own arrangement; Section 4.2 prints them for the
96-sphere snapshot. The full animation S4, with an interactive 3D version, plays in
[Lesson 03a, Section 4.1](../part3_refinement/03a_fixed_point.md).
```

### 4.2 Why the surrogate needs both a learned kernel and self-consistency

The toy lets us solve $(I + A)\,\mathbf s = \mathbf b$ exactly, because we know $G$ and therefore $A$.
A real simulation gives neither:

- **The kernel is unknown.** Real disturbances have wakes (Section 1.4), near-field structure that one slip
  vector cannot describe (Section 2.2), and they depend on the density ratio and on the crowd around the
  particle. The surrogate therefore *learns* the one-particle message from data. A0 keeps the structure we
  checked, a vector in $\mathrm{span}(\mathbf x_{fj}, \mathbf q_j)$ that is $O(3)$-equivariant, and replaces
  the fixed $\alpha$ and $\beta$ by networks of invariants (Part II).
- **The state is unknown.** The snapshot reports $\mathbf v_j$, not the slip. Superposing with the base
  state $\mathbf q^{\text{base}} = \mathbf v - \mathbf U_b$ (in the research, $\mathbf v - \langle\mathbf
  u_f\rangle$, Lesson 01c) is naive superposition, and Section 3 showed how badly that fails in a crowded
  suspension, in either frame. The fix is the toy's own condition, applied to the *model's* predicted flow
  instead of the exact one: find states with $\mathbf q = \mathbf q^{\text{base}} - W(\mathbf q)$, where
  $W_i$ is the model's prediction averaged over sphere $i$'s surface. That is the self-consistent
  refinement of Part III, solved by damped iteration because Section 4.1 showed that the undamped series
  can diverge.

The last cell runs that recipe on the toy with the exact kernel standing in for a trained model: start from
the base states and iterate the map $T(\mathbf q) = \mathbf q^{\text{base}} - A\mathbf q$ with
`course_utils.fixed_point.picard`, undamped and damped. Nothing here calls the toy's solver, so reaching
`snap.slip` is a real test.

```{code-cell} ipython3
from course_utils import fixed_point as fp

T_exact = lambda q: (q_base.reshape(-1) - A96 @ q.reshape(-1)).reshape(-1, 3)
mu = 1 + lam                                     # eigenvalues of I + A
print(f"eigenvalues of I + A in [{mu.min():.2f}, {mu.max():.2f}]: damping converges for alpha < {2 / mu.max():.2f}")
undamped = fp.picard(T_exact, q_base, alpha=1.0, tol=1e-13, max_iter=30)
damped = fp.picard(T_exact, q_base, alpha=0.5, tol=1e-13, max_iter=500)
print(f"undamped (alpha = 1):  residual {undamped.residuals[0]:.2e} -> {undamped.residuals[-1]:.2e} after "
      f"{undamped.n_iter} steps")
print(f"damped (alpha = 0.5): residual {damped.residuals[0]:.2e} -> {damped.residuals[-1]:.2e} after "
      f"{damped.n_iter} steps; step factor {np.abs(1 - 0.5 * mu).max():.2f}")
print(report("damped iteration from q_base reaches the toy's slips", np.abs(damped.q - snap.slip).max(), 1e-10))
print(report("field from the iterated states == truth", np.abs(superpose(damped.q) - true).max(), 1e-10))
print(f"exact kernel + base states: field error {field_errors(u_naive, true)[1]:.2f} (mean removed)")
print(f"lesson runtime so far: {time.time() - T_START:.0f} s")
```

With the exact kernel, the only thing separating naive superposition from the truth is the states, and a
damped iteration from the base states finds them, while the undamped one runs away.

## Summary

- One translating sphere in Stokes flow makes $G(\mathbf x)\mathbf s$: a $1/r$ Stokeslet that carries the
  drag plus a $1/r^3$ dipole that enforces no-slip. The toy multiplies it by $e^{-(r-a)/L}$, which keeps no-slip
  but breaks incompressibility and momentum balance (a stated toy simplification).
- The field is $\alpha\,\mathbf x + \beta\,\mathbf s$ with invariant coefficients, so it lies in
  $\mathrm{span}(\mathbf x, \mathbf s)$ and is $O(3)$-equivariant, both checked at round-off. Its fore-aft
  symmetry is special to Stokes flow and is lost when wakes form.
- Stokes flow is linear, so disturbances add. But each sphere must radiate its **slip**, its velocity
  relative to the fluid around it, not its own velocity.
- Naive superposition (each sphere radiates its base state $\mathbf v_j - \mathbf U_b$) breaks averaged
  no-slip. For a pair the error is largest near contact and larger in line than side by side. In a full
  toy snapshot most of it is one collective drift mode of the open box, which subtracting the measured
  mean $\langle\mathbf u_f\rangle$ from the states partly absorbs; the local remainder grows with the
  volume fraction until it is about as large as the fluctuation itself.
- The self-consistent slips fix the surface *average* but not the pointwise surface velocity. Reflections
  of the averaged system converge only when the spectral radius of the interaction matrix is below 1;
  damped iteration converges beyond that.
- The surrogate therefore learns the one-particle message (Part II) and finds the states self-consistently
  from its own predictions (Part III).

## Exercises

1. **Three in a row.** Place three spheres in line along $y$ with centre spacing $1.2\,d$ (surface gaps of
   $0.2\,d$), all moving with
   $(0, -1, 0)$, backflow off. Which sphere has the smallest slip, and why? Compare with the side-by-side
   arrangement.
2. **Why the toy screens.** Repeat the reflection experiment of Section 4.1 for $N = 16$ with
   `screening_length=None`. Print the spectral radius, and relate the result to the $1/r$ decay of Section 1.3.
3. **A uniform output offset cannot absorb the error.** Section 3.1 printed the naive field error with its
   mean removed. Show that no uniform vector $\mathbf c$ added to the naive *field* does better than
   removing the mean.

:::{admonition} Solution 1
:class: dropdown
```python
cfg3 = replace(cfg, n_particles=3, backflow=False)
v3 = np.tile([0.0, -1.0, 0.0], (3, 1))
for axis, name in ((1, "in line"), (0, "side by side")):
    off = np.zeros(3); off[axis] = 1.2
    sn = build_snapshot(centre + np.stack([-off, 0 * off, off]), v3, cfg3)
    print(name, "slip speeds:", np.linalg.norm(sn.slip, axis=1).round(3))
```
The middle sphere has the smallest slip in both arrangements. It sits in the flow of two neighbours, so the
fluid around it already moves down fastest, and it moves least relative to that fluid. The effect is
stronger in line, because a sphere's Stokeslet is twice as strong along its motion as across it.
:::

:::{admonition} Solution 2
:class: dropdown
```python
e, rho = reflections(make_snapshot(cfg, n_particles=16, seed=0, screening_length=None))
print(f"unscreened, N = 16: spectral radius {rho:.2f}; error after 12 reflections {e[12]:.1e}")
```
Even 16 spheres, a very dilute suspension, give a spectral radius above 1, so the series diverges. With
$1/r$ decay, every sphere feels every other sphere almost as strongly as its nearest neighbour, and the
interaction matrix has large eigenvalues for collective motion. Screening makes interactions local, keeps
the sums bounded in the open box and, at the course defaults, leaves a problem that a damped iteration can
solve (Part III). In a real sedimenting suspension the mean is set by the backflow and the mean pressure
gradient; how the fluctuations stay bounded is a subtler, long-debated question (Section 1.3). Either way it
is not an exponential factor.
:::

:::{admonition} Solution 3
:class: dropdown
```python
d = u_naive - true
c_best = -d.mean(0)                            # least-squares optimal uniform shift
for c, lab in ((np.zeros(3), "no shift"), (c_best, "best uniform shift"), (c_best + 0.05, "a different shift")):
    dd = d + c
    print(f"{lab:20s}: RMS error {np.sqrt((dd**2).sum(1).mean()):.4f}")
```
The RMS of $\mathbf d + \mathbf c$ is minimised by $\mathbf c = -\overline{\mathbf d}$ (set the derivative of
the mean square to zero), which is exactly removing the mean. So a uniform offset of the *output* cannot
remove the centred error. Changing the frame of the *states* is a different and larger effect: it changes
every radiated $G(\mathbf x - \mathbf r_j)\,\mathbf q_j$, not just the mean, and Section 3.2 showed that
using $\mathbf v - \langle\mathbf u_f\rangle$ instead of $\mathbf v - \mathbf U_b$ lowers the centred
error several-fold. Even then it does not reach zero.
:::

## References

- MolSSI e3nn course, [Introduction](https://molssi-ai.github.io/e3nn-course/book/intro.html): vectors,
  rotations, reflections and equivariance.
- {cite}`happel1983low`: the Stokes solution for a translating sphere and the method of reflections.
- {cite}`guazzelli2011physical`: hydrodynamic interactions, screening and hindered settling in suspensions.
- {cite}`lebedev1999quadrature`: the quadrature rule used for surface averages.
