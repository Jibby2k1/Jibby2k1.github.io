# Picard, damping and Anderson acceleration

:::{admonition} Coming soon
:class: note
The full notebook for this lesson is **coming soon**. The learning goals and outline below are fixed; the
notebook will follow the course template. Sections 1 and 2 are already here: two short derivations, each with
an animation. Until the rest arrives, the code for both solvers is in
[Lesson 03a](03a_fixed_point.md), Sections 4 and 5.
:::

## What you will learn

- run plain Picard and watch it diverge when the map is not a contraction
- explain damping alpha through the eigenvalues of the iteration matrix and choose alpha = 1/8
- implement Anderson acceleration and compare evaluations to convergence

## Outline

Failure first: undamped substitution diverges on the toy. Damping is analysed through the spectrum of the linear map, then Anderson acceleration is derived as a small least-squares problem over recent residuals and compared with Picard on residual curves.

## 1. Why damping works: one factor per mode

Refinement solves $\mathbf q = T(\mathbf q)$ with $T(\mathbf q) = \mathbf q^{\text{base}} - W(\mathbf q)$. Damped
Picard moves the state only a fraction $\alpha$ of the way to its target:

$$
\mathbf q_{k+1} = (1-\alpha)\,\mathbf q_k + \alpha\,T(\mathbf q_k).
$$

For the toy's exact map $W(\mathbf q) = A\mathbf q$, the error can be followed exactly. Let $\mathbf q^\star$ be
the fixed point and $\boldsymbol\varepsilon_k = \mathbf q_k - \mathbf q^\star$ the error. (Lesson 03a wrote
$\mathbf e$ for it; from here on the course uses $\boldsymbol\varepsilon$, because $\mathbf e_j$ is the state
direction.) Since $T(\mathbf q_k) - T(\mathbf q^\star) = -A\,\boldsymbol\varepsilon_k$ and
$T(\mathbf q^\star) = \mathbf q^\star$,

$$
\boldsymbol\varepsilon_{k+1} = (1-\alpha)\,\boldsymbol\varepsilon_k - \alpha A\,\boldsymbol\varepsilon_k
= \big(I - \alpha\,(I + A)\big)\,\boldsymbol\varepsilon_k .
$$

Expand the error in eigenvectors of $I + A$, with eigenvalues $\mu = 1 + \lambda$ where $\lambda$ is an
eigenvalue of $A$. Each **mode** evolves on its own, and one update multiplies it by one number:

$$
\text{factor}(\mu) = 1 - \alpha\,\mu .
$$

Plotted against $\mu$, the factor is a straight line through $(0, 1)$ with slope $-\alpha$. A mode shrinks when
its factor lies between $-1$ and $1$. On the default toy the eigenvalues $\mu$ are real and positive (the script
that prepares the animation's data asserts both), so the condition for every mode at once is

$$
-1 < 1 - \alpha\,\mu < 1 \ \ \text{for all } \mu
\qquad\Longleftrightarrow\qquad
0 < \alpha < \frac{2}{\mu_{\max}} ,
$$

and the error shrinks per update, asymptotically, by the worst factor
$\rho(\alpha) = \max_\mu\lvert 1 - \alpha\mu\rvert$. (Lesson 03a, Section 4.1 calls this number the contraction
factor $c(\alpha)$ and gives the condition for complex $\mu$. The clip below writes $\rho(\alpha)$, the usual symbol
for a spectral radius; it is not the density ratio $\rho$ of the model inputs.) Two things follow. With $\alpha = 1$ the factor is
$-\lambda$: every mode with $\lvert\lambda\rvert > 1$ flips sign and grows, which is the divergence of plain
substitution. With a very small $\alpha$ every factor is inside the band, but the slowest mode has factor
$1 - \alpha\mu_{\min}$, close to one, so convergence is safe and slow.

<div class="course-anim" style="margin:1.2em 0 0.4em">
<video controls playsinline preload="metadata" poster="../_static/media/R2_damping_eigenvalues/R2_damping_eigenvalues.png" aria-label="A state arrow slides part of the way to its target; one error mode is multiplied by one minus alpha mu for three values of alpha; a plot of that factor against mu, a straight line that pivots as alpha shrinks until every eigenvalue lies inside a band; error curves for the three alphas." style="display:block;width:100%;border-radius:14px;background:#0f1419;box-shadow:0 4px 18px rgba(0,0,0,.25)">
<source src="../_static/media/R2_damping_eigenvalues/R2_damping_eigenvalues.webm" type="video/webm">
<source src="../_static/media/R2_damping_eigenvalues/R2_damping_eigenvalues.mp4" type="video/mp4">
<track kind="captions" src="../_static/media/R2_damping_eigenvalues/R2_damping_eigenvalues.vtt" srclang="en" label="English">
Your browser cannot play this video. <a href="../_static/media/R2_damping_eigenvalues/R2_damping_eigenvalues.mp4">Download the MP4</a>.
</video>
</div>

**Animation R2, damping and eigenvalues.** A recap of the damped update for one particle, then one error mode by hand: $\boldsymbol\varepsilon \leftarrow (1 - \alpha\mu)\,\boldsymbol\varepsilon$ with $\mu = \mu_{\max}$ of the 96-sphere toy and $\alpha = 1$, $1/2$ and $1/8$. It flips and grows, shrinks fast, shrinks slowly. The main plot shows the factor $1 - \alpha\mu$ against $\mu$, with one tick for every eigenvalue of $I + A$ and the band $\lvert\text{factor}\rvert < 1$ shaded. At $\alpha = 1$ seven modes lie outside the band. As $\alpha$ decreases the line pivots about $(0, 1)$ toward flat, and the last mode enters the band at $\alpha = 2/\mu_{\max}$. The final panel runs the real iteration: $\lVert\boldsymbol\varepsilon_k\rVert/\lVert\mathbf q^\star\rVert$ diverges for $\alpha = 1$, falls fast for $1/2$ and crawls for $1/8$, and the measured late rates sit close to the predicted $\rho(\alpha)$. Notice the trade: a flatter line is safer for the fast modes and slower for the slow ones. For a learned map the eigenvalues are not known in advance, which is why the course keeps the cautious $\alpha = 1/8$. The clip is drawn on a dark stage in both page themes and has a voice-over with captions.

**Takeaway.** Damping does not move the fixed point; it rescales every error mode by $1 - \alpha\mu$. The
iteration converges exactly when all of those factors lie between $-1$ and $1$, and its speed is set by the worst
one. Choosing $\alpha$ is a trade between the largest $\mu$, which limits how big $\alpha$ may be, and the
smallest, which wants it big.

## 2. Anderson acceleration: use the last defects

Picard looks only at the current iterate. Its step follows the **defect**
$\mathbf f_k = T(\mathbf q_k) - \mathbf q_k$, which is zero exactly at the fixed point:
$\mathbf q_{k+1} = \mathbf q_k + \beta\,\mathbf f_k$, with the damping written $\beta$ here. When successive
defects point in nearly opposite directions, the iterates zigzag across the answer.

Anderson acceleration {cite}`walker2011anderson` uses the previous iterate too. With memory $m = 1$, look along
the line through the last two iterates,

$$
\tilde{\mathbf q}(\gamma) = \mathbf q_k - \gamma\,(\mathbf q_k - \mathbf q_{k-1}) .
$$

For a linear map the defect is an affine function of the state, so the defect at that point is the same
combination of the two known defects (for a nonlinear map this is an approximation):

$$
\tilde{\mathbf f}(\gamma) = \mathbf f_k - \gamma\,\Delta\mathbf f, \qquad \Delta\mathbf f = \mathbf f_k - \mathbf f_{k-1} .
$$

Choose the $\gamma$ that makes this defect as short as possible. Setting the derivative of
$\lvert\mathbf f_k - \gamma\,\Delta\mathbf f\rvert^2$ to zero gives one dot product over one squared length,

$$
\gamma = \frac{\mathbf f_k\cdot\Delta\mathbf f}{\lvert\Delta\mathbf f\rvert^2},
$$

and the update is a damped step from the mixed point along the mixed defect,
$\mathbf q_{k+1} = \tilde{\mathbf q}(\gamma) + \beta\,\tilde{\mathbf f}(\gamma)$. With memory $m$, keep the last
$m$ differences of iterates and of defects as the columns of $\Delta Q$ and $\Delta F$ and fit one coefficient for
each:

$$
\boldsymbol\gamma = \arg\min_{\boldsymbol\gamma}\big\lVert\mathbf f_k - \Delta F\,\boldsymbol\gamma\big\rVert,
\qquad
\mathbf q_{k+1} = \mathbf q_k + \beta\,\mathbf f_k - (\Delta Q + \beta\,\Delta F)\,\boldsymbol\gamma .
$$

This is the update of Lesson 03a, Section 5, and with $m = 0$ it is damped Picard with $\alpha = \beta$. The
least-squares problem has only $m$ unknowns, so its cost is negligible next to one evaluation of $T$.

<div class="course-anim" style="margin:1.2em 0 0.4em">
<video controls playsinline preload="metadata" poster="../_static/media/R3_anderson_acceleration/R3_anderson_acceleration.png" aria-label="Iterates zigzag toward a dashed green ring in a plane, each step following an orange defect arrow; a point slides along the line through the last two iterates to where the defect is shortest; Anderson steps from that point and gets much closer; residual curves on the course toy for Picard and for Anderson with memory one and three." style="display:block;width:100%;border-radius:14px;background:#0f1419;box-shadow:0 4px 18px rgba(0,0,0,.25)">
<source src="../_static/media/R3_anderson_acceleration/R3_anderson_acceleration.webm" type="video/webm">
<source src="../_static/media/R3_anderson_acceleration/R3_anderson_acceleration.mp4" type="video/mp4">
<track kind="captions" src="../_static/media/R3_anderson_acceleration/R3_anderson_acceleration.vtt" srclang="en" label="English">
Your browser cannot play this video. <a href="../_static/media/R3_anderson_acceleration/R3_anderson_acceleration.mp4">Download the MP4</a>.
</video>
</div>

**Animation R3, Anderson acceleration.** The first part uses a constructed linear map with a two-component state, so that states and defects can be drawn in one plane; it runs with $\beta = 1$, so each defect arrow (orange) reaches from an iterate to the next one. Picard follows the defect and zigzags toward the fixed point $\mathbf q^\star$ (dashed green ring): the first two defects point nearly opposite ways. A point then slides along the line through the last two iterates while a small plot shows the length of the interpolated defect; $\gamma$ marks its minimum, the least-squares solution above. Anderson moves to that point $\tilde{\mathbf q}$ and steps along its defect, and after four updates its error is far smaller than Picard's. The last part is the 96-sphere course toy with the exact map, the same damping $\alpha = \beta = 1/8$ for every solver and a tolerance of $10^{-10}$ on the residual $R$: Picard needs 268 updates, Anderson 155 with $m = 1$ and 111 with $m = 3$. Notice that the comparison is at equal damping. Anderson is not a bigger step; it is a better direction, built from defects that were already computed. The clip is drawn on a dark stage in both page themes and has a voice-over with captions.

**Takeaway.** $\boldsymbol\gamma$ is a least-squares fit: remember the last $m$ defects and extrapolate to where
they cancel. The fit is exact for a linear map and an approximation for a learned one, so the damping $\beta$
stays. Each update still costs one evaluation of $T$, one pass of the surrogate over every particle surface, and
that count is what Anderson reduces.

## References

- H. F. Walker and P. Ni, Anderson acceleration for fixed-point iterations {cite}`walker2011anderson`.
- [Lesson 03a](03a_fixed_point.md), Sections 4 and 5: both solvers in code on the same toy.
