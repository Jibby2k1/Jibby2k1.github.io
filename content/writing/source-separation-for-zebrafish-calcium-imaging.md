---
title: Source separation for zebrafish calcium imaging
slug: source-separation-for-zebrafish-calcium-imaging
description: What a broad experiment program taught us about separating neural activity from background, artifacts, and measurement noise without erasing the signal.
date: 2026-07-30
kind: research-journal
label: Research log
topics:
  - calcium imaging
  - source separation
  - signal processing
  - neuroengineering
heroImage: assets/img/research/source-separation/label-projection-overlay.webp
heroAlt: Zebrafish calcium-imaging field with the current annotated regions projected over the recording.
summary: The strongest result was not a single winning denoiser: preserving the activity carrier and using spatial separation features for ranking proved more reliable.
relatedProjects:
  - zebrafish-voltage-imaging-detection
takeaways:
  - Quiet-field standardization and spatial context contributed more than increasing model depth.
  - Source-separation outputs are most useful as bounded ranking evidence, not replacements for the scientific trace.
  - Sparse positive labels make exhaustive annotation the next important experiment.
---
The source-separation problem sounds straightforward: decompose an imaging movie into neural activity, background, artifacts, and measurement noise. In practice, the useful signal and the nuisance sources overlap in space, time, and frequency. A method can produce a visually clean movie while attenuating a real event, shifting its peak, or changing the trace that a later scientific analysis would interpret.

That makes the goal stricter than ordinary denoising. We are not only trying to make the movie look cleaner. We want to improve event discovery while preserving the amplitude, area, morphology, and timing of the activity itself.

## The experiment program

The recent calcium-burst program tested a broad set of hypotheses rather than betting on one model family. The screens included PCA and ICA representations, nonnegative factorization, local spectral Wiener filtering, nonlocal denoising, component-wise Parzen models, latent dynamics, morphology-aware spatial filters, CFAR variants, pairwise temporal separation, and learned candidate rankers.

The common real-data evaluation contains 79 known burst-specific observations across 27 region identities and four bursts. Those labels are sparse positives: a candidate that does not match one of them is unknown, not automatically a false positive. For that reason, fixed-budget recall—how many known events appear among the same number of candidates—is one of the fairest comparisons available today.

The breadth of the search is useful engineering evidence, but the runs are not thousands of independent scientific trials. Many models share the same recordings, labels, and held-out bursts. I treat the program as an architectural screen that eliminates weak assumptions and identifies the next experiment, not as a final leaderboard.

## What improved

At the primary budget of 58 candidates per burst, the archived centered-residual carrier recovered 51 of 79 known events. Quiet per-pixel standardization increased that to 55 of 79. Scoring the broader proposal union with the standardized carrier reached 56 of 79, a nested bounded-linear ranker reached 57 of 79, and leakage-safe single-feature selection reached 58 of 79.

That decomposition is more informative than the final number. Most of the measured improvement came from calibration and normalization. Proposal diversity contributed one additional match, and learned ranking contributed roughly one more. A residual multilayer perceptron tied the bounded linear model rather than beating it.

Spatial context was the most consistent useful information. Local spectral structure, cross-scale agreement, dense spatial ICA, center-versus-annular responses, and morphology-sensitive features repeatedly improved candidate ordering. Pure temporal derivatives were often visually informative, but they were weak or timing-distorting as standalone detectors.

![Known-event recall plotted against the number of candidates retained per burst.](../assets/img/research/source-separation/budget-recall-curves.svg "Held-out mean across four bursts. The standardized carrier leads at budgets 20 and 40; nested single-feature selection becomes strongest at 58 candidates and above.")

## Why the highest recall is not automatically the best detector

The ranking result changes at tighter candidate budgets. At budgets of 20 and 40, the native standardized carrier still recovered more known events than the learned rankers. The learned methods helped after the list expanded to 58 or more candidates, but they did not yet improve the very top of the list.

That distinction prevents an easy overclaim. A method that recovers more labels while emitting hundreds of candidates has not demonstrated better precision. Candidate burden measures how much a scientist must review; it does not tell us which unmatched candidates are real unlabelled neurons and which are artifacts.

The same caution applies to visual quality. Parzen innovation backgrounds and several source-separation lanes can produce convincing remainders while scoring below the raw carrier at a calibrated threshold. Conversely, a high-recall method may distort signal timing or flood the review queue. A trustworthy promotion needs both visual inspection and a preregistered operating point.

![Ranked candidate overlay from the nested linear model on the first held-out burst.](../assets/img/research/source-separation/ranked-candidate-overlay-burst1.webp "A v5 diagnostic overlay showing the candidate field produced for one held-out burst. Candidate burden is visible evidence for review effort, not a precision estimate.")

## What the denoising screens taught us

No denoising family passed every preservation gate. Different methods occupied different parts of a Pareto frontier:

- Local PSD-Wiener filtering produced strong known-label recall and noise attenuation, but increased candidate burden and missed the timing and synthetic-correlation gate.
- Nonnegative factorization was one of the most balanced separation leads, preserving event amplitude while improving recall, but it still needed better component selection.
- Nonlocal patch denoising preserved real-event amplitude almost exactly and reduced quiet energy, but did not improve synthetic recovery.
- Component-wise Kalman filtering reduced noise and improved synthetic correlation, but shifted the synthetic peak.
- Per-component Parzen ICA produced strong fixed-budget recall, but attenuated real peaks beyond the frozen tolerance.
- Dense spatial ICA with Wiener reconstruction improved over a coarse patch lattice, but the complete preservation audit still stopped promotion.

These are useful failures. They show that noise reduction, event preservation, and candidate ranking are related objectives, but they are not interchangeable.

![Side-by-side diagnostic comparison of source-separation and denoising outputs at frame 212.](../assets/img/research/source-separation/denoising-audit-frame212.webp "Frame 212 from the sequential denoising audit. The comparison is used to inspect whether a cleaner-looking output also preserves localized neural structure.")

## The architecture that survived

The recurring safe design is to preserve an immutable activity carrier, compute source-separation and denoising channels as auxiliary evidence, and use a separate score for candidate ranking.

The carrier remains the trace used for scientific interpretation. Auxiliary channels can say that a candidate resembles a local subspace, a center-shaped source, a membrane-like structure, a cross-scale consensus peak, or a persistent artifact. They may change where a candidate appears in the review queue, but they do not silently replace the underlying signal.

This also reveals two different failure modes. Some known events are missing from the entire proposal union; those need a new representation or morphology expert. Other events are proposed but ranked below the candidate budget; those need better ranking or hard-negative discrimination. One end-to-end recall number hides that difference.

## The next experiment is better annotation

The main bottleneck is now measurement rather than model capacity. Sparse positive labels cannot support ordinary precision estimates or reliable hard-negative learning. The highest-value next step is therefore a bounded field in which every visible candidate is reviewed as neuron, background, artifact, or unresolved.

For neural events, the annotation should also capture center-versus-membrane appearance, isolated-versus-crowded context, footprint, burst activity, onset confidence, and persistent structure. That single effort would enable real precision-recall analysis, morphology-conditional evaluation, and a clean proposal-versus-ranking ablation.

Only after that checkpoint should a more complex mixture of experts be promoted. The advancement rule is deliberately strict: improve precision-aware detection at useful candidate budgets, preserve the carrier's timing and amplitude, remain stable across region identities, and retain a plausible causal implementation path.

The current result is therefore not that source separation has been solved. It is that the problem is now better specified. Spatial context matters, calibration matters, deeper models are not automatically better, and the scientific signal needs a protected path through the system.
