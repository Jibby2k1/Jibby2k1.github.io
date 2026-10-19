---
title: Redesigning a model so its latents count
slug: redesigning-a-model-so-its-latents-count
description: Three pre-registered redesigns of a hierarchical linear dynamical system, a label-free gate on whether its latent layers had stopped collapsing, and what happened when two of them reached a labelled screen.
date: 2026-10-18
kind: research-journal
label: Research log
topics:
  - state-space models
  - preregistration
  - representation learning
  - research methodology
heroImage: assets/img/research/hlds/redesign-participation.webp
heroAlt: Left, dot plot of the participation ratio of two latent layers under four model designs, rising from about 1 to about 5 for one layer; right, accuracy above the null for the input and the two layers in one design, where the input is highest on every row.
summary: Giving the model a richer input and fitting it by expectation-maximization (EM) made its upper layers use more dimensions, one of them about five instead of one, but in the labelled explore screen neither layer decoded the three conditions as well as the input it was built from.
relatedProjects:
  - latent-signal-dynamical-classification
takeaways:
  - A latent layer that uses more dimensions is not a layer that knows more; a label-free gate can show the collapse is gone, and only a labelled comparison against the input and a matched control can say whether anything new arrived.
  - Freezing the menu, the gate and the branch rules before any run meant a failed gate ended a design instead of starting a tuning session.
  - The strongest result of this phase was about the input, not the model, and it stays a lead until it has its own pre-registration and its one held-out test.
---
The [first phase](does-a-hidden-layer-know-more-than-its-input.html) of this project asked whether the latent layers of a hierarchical linear dynamical system (HLDS) carry condition information beyond their own input, on multichannel time-series recordings from a collaborator's dataset with three condition labels. None of seven labelled configurations had shown a significant gain over its input, and none of 38 label-free knob settings had stopped the upper layers collapsing. This post is about changing the model rather than its knobs.

## Why the layers collapsed

In the default model both upper layers, u (16 dimensions) and z (2), sat at a participation ratio of about 1.1: the effective number of dimensions a layer uses within a trial. A label-free diagnosis on two recordings pointed first at the input. The input was 64 columns, but seven principal components held 99.4 to 99.7 percent of its variance, and its slow content was essentially one direction: the lag-average of the lowest frequency band, after summing all channels into one. The dominant latent mode was that direction.

The second cause was inside the model. A norm clip held u's own transition matrix at a norm of 0.97 while its spectral radius was only 0.51 to 0.85, so u's own modes died out within a few samples and u became a filtered copy of z. The diagnosis also showed my earlier reports had the training rule wrong (Adam, not recursive least squares) and that z was a random walk with no learned dynamics; I filed an erratum.

The one lever that moved anything was a richer input. Keeping four spatial principal components instead of summing the channels raised the low-frequency participation ratio of the input from 1.2–1.3 to 4.3–5.0.

## The menu and the gate

The new question, frozen before any run: once the input has more than one slow direction, does a redesigned layer carry condition information that is easier to read than its own input and than an unsupervised projection of that input with the same number of dimensions? Three designs, in a fixed order:

- **D1:** four spatial components × four frequency bands × four contiguous lags (64 columns), with the inverted noise setting from the first phase (σx 0.01, σu and σz 0.1): small process noise on the bottom latent layer x and larger noise on u and z, the reverse of the default ordering.
- **D2:** D1 with a 4-dimensional z, and u split into four blocks, one per spatial component, with u's own dynamics and its coupling into the bottom layer block-diagonal.
- **C:** the 16-lag spatial input, reduced by a training-fold PCA within each component-and-band block to two whitened columns (32 in all), and fitted by 60 iterations of expectation-maximization (EM), with diagonal dynamics in the bottom layer, u's coupling into that layer structured by band, and learned dynamics for z.

Each design had to pass a label-free move gate (the first phase's move check, now required on two folds) on two recordings before any label was touched: u or z with a participation ratio of at least 1.5 and a band-power spread (between-trial over within-trial variation) of at least 0.35, on one fold and again on a second. D2 also needed u to use more dimensions than z, or u would just be inheriting z's rank. C needed its EM likelihood never to decrease and z's learned dynamics to stay stable.

Calibrating the gate was sobering: stationary noise with no trial differences scores a spread of 0.38 to 0.42. A move shows the collapse has gone, not that anything new arrived. So the success rule kept the earlier screen and one held-out confirm run, and added a control. The screen looks at a layer's best row, its best pairing of feature summary and probe as ranked by the gap to the input: that gap must be at least zero, and the row's own delta (accuracy above the design null, in which each draw relabels a recording's trials with another recording's real condition sequence) must be positive at p < 0.05. The control: that best row had to beat the log-variance of a PCA of the input with the layer's own dimension, fitted on each fold's training trials only. Each new input also had to pass an input gate, run only after its design's move gate so that gate stayed blind to labels: a positive mean gain over the channel-summed input on its best row.

## What happened

![Left: participation ratio of u and z for the default model and designs D1, D2 and C on the two gate recordings. Right: accuracy above the design null for the input, u and z on covariance, band-power and log-variance rows in design C.](../assets/img/research/hlds/redesign-participation.webp "Left: within-trial participation ratio on the two gate recordings; one dot per fold for D1, D2 and C, a five-fold mean for the default. The gate also needed a band-power spread of at least 0.35, which D1, D2 and C met and the default did not. z has 2 dimensions except in D2 (4). Right: design C's labelled explore screen, 14 recordings, logistic-regression probe, mean with t-based 95% intervals (±2.16 standard errors). Layer log-variance rows were reported but never counted toward a pass.")

**D1** moved: u 1.91 and 1.98, z 1.79 and 1.71 on the two folds. Only the second fold was blind, since the first had come from the design pilot. Its input passed the input gate narrowly (+0.017 over the channel-summed input, p 0.57). On the labelled explore screen (14 recordings, five folds of contiguous trial blocks) the best u row was 0.013 below the input in balanced accuracy, its own delta was +0.012 (p 0.34), and it was 0.034 below the matched 16-dimensional PCA control (p 0.13). z was 0.014 below the input. The layers had stopped collapsing and still showed no gain over their input.

**D2** failed its gate. The block masks lifted u from 2.17 (the same model with a 4-dimensional z but no masks, one fold) to about 2.8, but z reached 3.27 and 3.05, so u stayed below z in both folds. It never got a labelled number. The failure points at the norm clip; the frozen rules allowed a fix only through a new dated addendum to the pre-registration, and I did not write one.

**C** moved u the furthest: 5.26 and 5.14, about five dimensions, against about one for the default and two for D1. u held up across all 14 explore recordings (participation ratio about 6), but z was marginal at 1.52 and 1.55 and fell to 1.44 across the 14, so z's move is a statement about two recordings. A reported-only run with unstructured EM (one fold, never gated) moved as much or more, u 5.74 and z 1.95, so most of the move came from EM on the reduced input, not from the structure I had designed.

In the labelled explore screen, C's best u row was 0.004 below the input (own +0.008, p 0.64) and z's was 0.026 below. One u row, the covariance of u under a shrinkage linear discriminant analysis (LDA) probe, was the first non-log-variance u row of any round so far to clear p < 0.05 on its own (+0.036, p 0.016 uncorrected, 0.062 after false-discovery correction across the run; the logistic-regression version plotted here had p 0.068), but it was not a best row and was still 0.023 below the input's covariance row. The input's log-variance row was the strongest of the run, +0.081 (p 0.0004), the largest input-level own delta of any round so far, and its covariance row gave +0.063 (p 0.003). C decompressed u, but what the layers held was a harder read of the input, not an easier one.

## Where that left things

Every design had been through its gate: D2 failed it, and D1 and C passed it but failed the explore screen, so none came near a confirm run and the negative stop fired after about 3.3 hours of a 20-hour compute cap. No confirm run was used; the 13 held-out recordings stayed untouched. Nothing above is confirmatory, and the +0.081 was an input row reported for context, not one the success rule could count.

Untested: a redesign of the norm clip that capped u, a two-rate model on a log-envelope input, and C's sensitivity to its initialisation, seed and noise floor. The next phase changed how the model meets the labels, in two directions: the HLDS as a generative model of each condition, with a static per-condition Gaussian among the competitors, in [when the static model is the honest baseline](when-the-static-model-is-the-honest-baseline.html), and a label channel on a log-envelope input (without the two-rate model), whose synthetic gate has its own post, [a gate that could not fail for the right reason](a-gate-that-could-not-fail-for-the-right-reason.html). The project overview is on the [project page](../projects/latent-signal-dynamical-classification.html).
