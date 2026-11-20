# Speed: layout, compile and TF32

:::{admonition} Coming soon
:class: note
This lesson is **coming soon**.
The learning goals and outline below are fixed; the notebook will follow the course template.
:::

## What you will learn

- profile A0 and see where time goes
- use component-major layouts, torch.compile, and understand what TF32 does to accuracy

## Outline

Practical performance notes: why the plain-PyTorch A0 is faster than the e3nn one, when torch.compile helps, and which computations must not run in TF32 (dot products in pair invariants).
