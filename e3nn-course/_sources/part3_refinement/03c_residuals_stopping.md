# Residuals and stopping

:::{admonition} Coming soon
:class: note
The full notebook for this lesson is **coming soon**. The learning goals and outline below are fixed; the
notebook will follow the course template. Section 1 is already here: a short derivation with an animation.
Until the rest arrives, the residual convention and three stopping rules are in
[Lesson 03a](03a_fixed_point.md), Sections 6 and 7.
:::

## What you will learn

- define the residual ||T(q) - q|| / ||q|| and explain why it is independent of damping
- relate residual to state error and to field error on the toy
- choose a fixed budget versus a tolerance, and recognise when a lower residual does not mean a better field

## Outline

Defines the course's residual convention, plots residual, state error and field error together, and discusses stopping by tolerance or by a fixed number of updates when the model itself is imperfect.

## 1. Converged is not the same as correct

Two different questions hide behind "is the refinement done?".

*Has the iteration converged?* The residual answers this. It is computed with the map that is actually being
iterated, here a model's map $\hat T$:

$$
\hat R(\mathbf q) = \frac{\lVert\hat T(\mathbf q) - \mathbf q\rVert}{\lVert\mathbf q\rVert} .
$$

It asks how far one more undamped update would move the state, it needs no reference data, and it is zero
exactly at the model's fixed point $\hat{\mathbf q}^\star = \hat T(\hat{\mathbf q}^\star)$.

*Is the state right?* The error answers this,

$$
E_k = \frac{\lVert\mathbf q_k - \mathbf q^\star\rVert}{\lVert\mathbf q^\star\rVert},
$$

and it needs the true state $\mathbf q^\star$, which is known in the toy and unknown in practice.

The two agree only if the model's map has the true fixed point. Take linear maps to see by how much they can
differ: the exact $T(\mathbf q) = \mathbf q^{\text{base}} - A\,\mathbf q$ and an imperfect
$\hat T(\mathbf q) = \mathbf q^{\text{base}} - \hat A\,\mathbf q$. Both fixed points satisfy a linear system with
the same right-hand side, $(I + \hat A)\,\hat{\mathbf q}^\star = \mathbf q^{\text{base}} = (I + A)\,\mathbf q^\star$.
Subtract $(I + \hat A)\,\mathbf q^\star$ from both sides:

$$
\hat{\mathbf q}^\star - \mathbf q^\star = -\,(I + \hat A)^{-1}(\hat A - A)\,\mathbf q^\star .
$$

The offset is the model's interaction error applied to the true state. No amount of iteration removes it,
because the iteration is solving the model's equation. For two equal spheres in line (Lesson 03a, Section 3, with
coupling $t$) and a model whose coupling is 30 % too strong, the truth is $q^\star = 1/(1+t)$ and the model's
fixed point is $\hat q^\star = 1/(1 + 1.3\,t)$, so $\hat q^\star < q^\star < q^{\text{base}} = 1$. The damped
iterates start at 1, slide down, **pass the truth** and settle below it. Somewhere on the way the error was
smaller than at the end. The best update count $k^\star = \arg\min_k E_k$ is then an interior point, while
$\hat R$ keeps falling.

<div class="course-anim" style="margin:1.2em 0 0.4em">
<video controls playsinline preload="metadata" poster="../_static/media/R4_model_fixed_point_vs_truth/R4_model_fixed_point_vs_truth.png" aria-label="A dot hops along a number line past a white ring, the truth, and stops on a green ring, the model's fixed point; a trail of iterates in a plane approaches the white ring and then curls away to the green one; a dashed residual curve keeps falling while a solid error curve turns around; a held-out curve picks the number of updates." style="display:block;width:100%;border-radius:14px;background:#0f1419;box-shadow:0 4px 18px rgba(0,0,0,.25)">
<source src="../_static/media/R4_model_fixed_point_vs_truth/R4_model_fixed_point_vs_truth.webm" type="video/webm">
<source src="../_static/media/R4_model_fixed_point_vs_truth/R4_model_fixed_point_vs_truth.mp4" type="video/mp4">
<track kind="captions" src="../_static/media/R4_model_fixed_point_vs_truth/R4_model_fixed_point_vs_truth.vtt" srclang="en" label="English">
Your browser cannot play this video. <a href="../_static/media/R4_model_fixed_point_vs_truth/R4_model_fixed_point_vs_truth.mp4">Download the MP4</a>.
</video>
</div>

**Animation R4, the model's fixed point is not the truth.** The imperfect map is constructed, not trained: $\hat T(\mathbf q) = \mathbf q^{\text{base}} - 1.3\,A\,\mathbf q$, the exact toy map with its interaction made 30 % too strong, as a stated stand-in for a learned model. First the two-sphere case above with $\alpha = 1/2$: the iterate passes the truth (white ring) and stops on the model's fixed point (green ring). Then the 96-sphere toy with $\alpha = 1/8$ and 60 updates, its 288 state components projected onto the plane through $\mathbf q^{\text{base}}$, $\mathbf q^\star$ and $\hat{\mathbf q}^\star$: the trail approaches the truth, then curls away toward the green ring. The plot draws the residual $\hat R$ (dashed) and the error $E_k$ (solid) against the update count on a log axis. The residual falls at every update; the error turns around at update 7. Notice the two readouts: at the best update the residual is still large, and after 60 updates the residual is tiny while the error is larger. The last part chooses the number of updates on a held-out snapshot, where the truth is available. Its choice, 12 updates, applied to the first snapshot gives a smaller error than running to convergence. The clip is drawn on a dark stage in both page themes and has a voice-over with captions.

**Takeaway.** The residual measures convergence to the *model's* fixed point,
$\hat{\mathbf q}^\star = \hat T(\hat{\mathbf q}^\star) \ne \mathbf q^\star$, and says nothing about the distance
from there to the truth. A lower residual is therefore not always a better state. When the model is imperfect,
treat the number of updates as a setting and choose it where the error on held-out data is lowest. Lesson 03a,
Section 7 meets the same gap with a trained mini-A0: both solvers converge to the model's fixed point, which is
closer to the exact slip than the base state but not equal to it.
