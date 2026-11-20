# Picard, damping and Anderson acceleration

:::{admonition} Coming soon
:class: note
This lesson is **coming soon**.
The learning goals and outline below are fixed; the notebook will follow the course template.
:::

## What you will learn

- run plain Picard and watch it diverge when the map is not a contraction
- explain damping alpha through the eigenvalues of the iteration matrix and choose alpha = 1/8
- implement Anderson acceleration and compare evaluations to convergence

## Outline

Failure first: undamped substitution diverges on the toy. Damping is analysed through the spectrum of the linear map, then Anderson acceleration is derived as a small least-squares problem over recent residuals and compared with Picard on residual curves.
