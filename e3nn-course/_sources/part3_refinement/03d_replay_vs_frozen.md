# Replay versus frozen-model inference

:::{admonition} Coming soon
:class: note
This lesson is **coming soon**.
The learning goals and outline below are fixed; the notebook will follow the course template.
:::

## What you will learn

- distinguish training-time state updates from inference with one frozen model
- implement replay (one update per checkpoint) and frozen inference (k updates with the final model)
- explain why the two can score differently and which one a paper should report

## Outline

Trains a few toy cycles where states are updated between training rounds, then evaluates a held-out snapshot two ways: replaying the training-time sequence of checkpoints, or iterating the final model alone. Shows the two states differ and argues for reporting the procedure a user can actually run.
