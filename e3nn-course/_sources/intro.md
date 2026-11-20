# Equivariant surrogates for particle-laden flow

*A hands-on companion course to the u1e3nn project.*

When many particles settle through a fluid, each one drags the fluid around it, and every particle feels the
flow made by all the others. Particle-resolved direct numerical simulation (DNS) resolves this flow exactly, but
it is expensive. This course teaches, step by step, how to build a fast neural **surrogate** for it: a model
that predicts the fluid velocity at any point from the positions and states of the nearby particles. It also
shows how to make the model respect the symmetries of physics exactly, and how to make the particle states
agree with the flow the model predicts.

You will build three things:

1. **The A0 model**, an E(3)-equivariant network that sums one *message* per neighbouring particle. Each message
   is built from the offset to that particle and the particle's state direction, and lies in the plane of those
   two vectors.
2. **Self-consistent refinement**: a damped fixed-point iteration (Picard, Anderson) that updates each particle's
   state using the model's own predicted flow, averaged over the particle's surface.
3. **Model B**, an invariant transformer that lets neighbours attend to each other. It can represent
   *shielding*, where one particle changes the effect of another, which no sum of independent pairs can do.
   On the way there is a short, self-contained **introduction to transformers**: queries, keys and values,
   softmax, multiple heads, masking, and why attention suits sets of particles.

## A companion to the MolSSI e3nn course

This course is modelled on, and builds directly on, the
[MolSSI e3nn course](https://molssi-ai.github.io/e3nn-course/book/intro.html) {cite}`molssi_e3nn_course`. That
course teaches the foundations of Euclidean neural networks {cite}`geiger2022e3nn`: groups and representations,
irreps, spherical harmonics, tensor products and the e3nn library. **We do not re-teach those topics.** When a
lesson needs one, it links to the MolSSI lesson that explains it. Read it there first if the idea is new to you.

What this course adds:

- **A complete scientific application**: a surrogate for the flow around settling particles, from the physics of
  one sphere to a trained model, with the decisions and pitfalls of a real research project.
- **Vectors only, done thoroughly.** Our models need only scalars (`0e`) and vectors (`1o`). This keeps the e3nn
  part small, and lets us show exactly what e3nn computes by writing the same network in plain PyTorch.
- **Fixed-point refinement**: treating particle states as unknowns that must agree with the model's prediction,
  and solving for them with damped and accelerated iterations.
- **Attention and transformers from scratch**, then a geometric, exactly invariant version of attention.
- **Evaluation practice**: splitting by snapshot, choosing between NRMSE and R², paired comparisons across seeds,
  and reading negative results.

## How the course teaches

We keep the MolSSI course's teaching philosophy and adapt it to this application:

- **Theory first, then code, side by side.** Each idea is stated in equations and then implemented in a few lines
  you can run and change.
- **Every symmetry claim is checked numerically.** If a lesson says a function is equivariant, invariant or
  permutation-invariant, the notebook verifies it on random inputs with random rotations *and reflections*, in
  float64, and prints the error (typically about $10^{-15}$). The shared helpers live in
  `course_utils.equivariance`.
- **Failure before success.** Each new piece is introduced by showing what goes wrong without it: a bias that
  breaks equivariance, an iteration that diverges without damping, a sum of pairs that plateaus when particles
  shield each other.
- **Many pictures.** Fields, particles, attention weights and residual curves are plotted throughout. Figures
  use transparent backgrounds and mid-tone colours, so they read in both the light and dark themes.
- **Short lessons.** Each part is split into lessons a, b, c, ... Every lesson has *What you will learn*,
  prerequisites, numbered sections, a summary, exercises with collapsed solutions, and references.

## A toy problem you can run anywhere

The research behind this course used particle-resolved DNS of settling suspensions, provided by collaborators.
That data is not part of the course, and the lessons quote no numbers from it. Instead, every lesson generates its
own **toy suspension** (`course_utils.toy_data`). It uses the exact Stokes field of a translating sphere, a
self-consistent slip velocity for every particle, a uniform backflow, and optional screening and three-body
shielding. The toy keeps the structure the models must learn and states its simplifications honestly. Every
notebook runs on a CPU in a few minutes.

## Outline

| Part | Topic |
|---|---|
| 0 | Setup and a first equivariance check |
| I | The problem: settling particles, superposition of one-particle disturbances, reference frames |
| II | The A0 model: pair geometry, scalar and vector branches, the span(x, e) property, e3nn vs plain PyTorch |
| III | Self-consistent refinement: fixed points, Picard, damping and Anderson, residuals, replay vs frozen inference |
| IV | Training and evaluation: splits, NRMSE vs R², seeds and paired comparisons, a worked toy case |
| V | Beyond pairwise: shielding, a mini introduction to transformers, invariant attention (model B), negative results |
| VI | Practice: speed, profiling, running on a cluster |

The first lessons are being written in batches. Pages marked *coming soon* already list their learning goals.

## Running the course

See the `README.md` in the `course/` folder. In short: install `course/requirements.txt` (a CPU build of PyTorch
is enough) and run `jupyter-book build course/`. Each lesson is a MyST-Markdown notebook that you can also open
directly in JupyterLab.
