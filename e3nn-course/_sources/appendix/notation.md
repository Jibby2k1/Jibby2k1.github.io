# Notation

| Symbol | Meaning | Code name |
|---|---|---|
| $N$ | number of particles in a snapshot | `snap.n` |
| $a$, $d = 2a$ | particle radius and diameter (toy default $a = 0.5$, $d = 1$) | `config.radius` |
| $\mathbf r_j$ | centre of particle $j$ | `snap.positions` |
| $\mathbf v_j$ | lab-frame particle velocity | `snap.velocity` |
| $\langle \mathbf u_f\rangle$ | measured mean fluid velocity over exterior points (the reference frame) | `snap.exterior_mean()` |
| $\mathbf U_b$ | toy backflow $-\phi/(1-\phi)\,\langle\mathbf v\rangle$ | `snap.backflow` |
| $\mathbf q_j$ | particle state (slip relative to the surrounding fluid); $\lvert\mathbf q_j\rvert$ plays the role of the particle Reynolds number | `states`, `q` |
| $\mathbf q^{\text{base}}$ | state known before refinement, $\mathbf v - \langle\mathbf u_f\rangle$ | `q_base` |
| $\mathbf s_j$ | the toy's exact self-consistent slip | `snap.slip` |
| $M$ | number of nearest neighbours a query point sees | `M` |
| $\mathbf x_{fj}$ | offset from the centre of neighbour $j$ to the query point $\mathbf x_f$ | `x_fj` |
| $\mathbf e_j$ | state direction $\mathbf q_j/\lvert\mathbf q_j\rvert$ | `e_j` |
| $\rho$ | particle-to-fluid density ratio | `scalars[..., 1]` |
| $\phi_j$ | local volume fraction around particle $j$ | `scalars[..., 2]` |
| $C$ | width (channels) of A0's branches | `width` |
| $y_u$, $\mathbf v_u$ | scalar-branch and vector-branch channel $u$ | `scalar_branch`, `vector_branch` |
| $\mathbf m_j$ | message from neighbour $j$; the output is $\sum_j \mathbf m_j$ | `messages()` |
| $W_i(\mathbf q)$ | model-predicted flow around particle $i$, averaged over its surface | `refinement.induced_velocity` |
| $T(\mathbf q) = \mathbf q^{\text{base}} - W(\mathbf q)$ | state map; refinement solves $\mathbf q = T(\mathbf q)$ | `refinement.picard_map` |
| $\alpha$ | Picard damping, $\mathbf q \leftarrow (1-\alpha)\mathbf q + \alpha T(\mathbf q)$ | `alpha` |
| $R(\mathbf q)$ | residual $\lVert T(\mathbf q)-\mathbf q\rVert/\lVert\mathbf q\rVert$ | `fixed_point.fp_residual` |
| $P_{jk}$ | pair invariants of neighbours $j,k$ used by model B | `models.pair_invariants` |
| $\alpha_j, \beta_j$ | model B's invariant output coefficients, $\mathbf u = \sum_j \alpha_j \mathbf x_{fj} + \beta_j \mathbf e_j$ | `MiniB.coefficients` |

Irreps follow e3nn: `0e` = scalar, `1o` = polar vector, `1e` = pseudovector. In code, vectors are always
Cartesian `(..., 3)` arrays, and a group element $g$ acts as `v @ g.T`.
