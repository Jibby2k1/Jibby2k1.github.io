---
title: Why I care about state-space models for EEG
slug: why-i-care-about-state-space-models-for-eeg
description: A quick note on using HLDS and related models to capture structure in EEG, instead of treating everything as i.i.d. features.
date: 2025-11-01
kind: research-journal
label: Research log
topics:
  - eeg
  - state-space models
  - signal processing
heroImage: assets/img/projects/HLDS_Inf.png
heroAlt: Diagram representing a state-space modeling pipeline for EEG.
summary: EEG work gets more informative when temporal structure is part of the model instead of something the pipeline averages away.
relatedProjects:
  - time-series-ml-experiment-infra
takeaways:
  - Temporal dependence is part of the signal, not only a nuisance to remove.
  - Latent-state models make debugging and interpretation easier than static feature pipelines.
  - Good research tooling matters because the value of a state-space model depends on reproducible preprocessing and evaluation.
---
EEG is genuinely dynamical. Adjacent windows are not independent, latent state matters, and a pipeline that ignores temporal structure usually pushes the hardest scientific question out of the model and into post-hoc interpretation.

State-space models force a cleaner separation between what is observed, what is hidden, and what is drifting over time. That matters when the goal is not only prediction but also understanding whether a result is stable across subjects, sessions, and recording conditions.

I keep returning to HLDS-style models because they give me a vocabulary for debugging. If the learned state is unstable, the issue may be preprocessing, segmentation, artifact handling, or model mismatch. That is more informative than a black-box score moving up or down without any clue about what changed.

For EEG research at UF, this translates into practical choices: cleaner windowing, clearer diagnostics, and a better bridge between mathematical structure and the real biological variability that shows up in lab data.
