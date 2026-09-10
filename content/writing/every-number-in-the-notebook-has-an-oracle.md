---
title: Every number in the notebook has an oracle
slug: every-number-in-the-notebook-has-an-oracle
description: The verification rule behind the SPS curriculum, and what happened when a manifold optimization demo failed its own plot title.
date: 2026-09-08
kind: research-journal
label: Teaching
topics:
  - teaching
  - numerical verification
  - optimization
  - jupyter
heroImage: assets/img/blog/every-number-in-the-notebook-has-an-oracle.svg
heroAlt: Diagram of a numerical claim checked against an independent oracle, with one branch shipping the result and one branch narrating the failure.
summary: A workshop number only counts once something the author did not derive agrees with it; when it does not, the notebook says so instead of hiding it.
relatedProjects:
  - sps-curriculum-workshops
takeaways:
  - A numerical claim in a teaching notebook is checked against an oracle the author did not hand-derive, such as a closed form, a reference solver, brute force, or a planted truth.
  - When a demo fails its check, the honest move is to narrate the failure and diagnose it, then fix the experiment only when the fix can be re-run and re-verified.
  - Riemannian optimization fixes the constraint, not the conditioning, and the demo only taught that once it was allowed to fail in public.
---
The workshop library I maintain for the IEEE Signal Processing Society (SPS) student chapter at the University of Florida holds 87 teaching notebooks across 12 topics, cut into 258 sessions of 30 to 40 minutes. What makes them usable is not the topic list but one rule in the contributing guide: a key numerical claim must be checked against an **independent oracle**, something the author did not hand-derive. A closed form. A reference solver such as LAPACK through `np.linalg.eigh`. An exhaustive brute-force search. A planted ground truth the demo is supposed to recover.

The rule ends with the sentence I care about most: if a demo's numbers contradict its prose, fix the experiment or the prose, never ship them disagreeing.

## What an oracle looks like in practice

The checks are small and specific, and the committed outputs are the evidence. The quantum workshop builds a quantum Fourier transform from gates and prints the largest entry of its difference from the discrete Fourier transform matrix: 3.78e-15. The channel coding workshop does not show that Viterbi decoding works; it shows that Viterbi reproduces exhaustive maximum-likelihood decoding exactly, so the bit-error curve measures the code, not the decoder. The convex optimization workshop checks that ADMM and FISTA recover the same sparse support, a stronger claim than fourteen decimals of agreement. The distributed training workshop compares a gradient averaged across four workers against the single-batch gradient: 8.94e-08, at the level of float32 machine epsilon. The capstone plants 55 bursts into synthetic IQ data and the CFAR detector finds 55. What they buy is a notebook a student can run alone and trust.

## What happens when the check is missing

I learned the rule's second half the hard way in July. Session 1 of the Manifold Optimization workshop minimizes a Rayleigh quotient on the unit sphere by Riemannian gradient descent: project the gradient onto the tangent space, step, retract by normalizing. The plot title read "descent on the sphere → the bottom eigenvector, natively." Session 2, the Stiefel-manifold version, ends with an `assert` against `eigh`; Session 1 had none.

On 2026-07-24, during the narration pass described below, I read the committed output properly. The final Rayleigh quotient was 0.5015. The true minimum eigenvalue, printed by `eigh` on the very next line, was 0.0097. The run was about 52 times off its target, with eigenvector alignment 0.533 where success is about 1.0. The output contradicted the title.

The diagnosis was more useful than the catch. The matrix was `S = M @ M.T` with `M` a square Gaussian matrix, which makes `S` Wishart. By the Marchenko–Pastur law its eigenvalue density blows up like one over the square root of x near zero, so the smallest eigenvalues are packed together. Rayleigh-quotient descent converges at a rate set by the relative gap: the distance from the smallest eigenvalue to the second smallest, divided by the width of the spectrum. Here that gap is essentially zero, and raising the iteration count would have been the wrong fix: the gap was the problem, not the budget.

## Narrate the failure, then fix it properly

I had no NumPy environment in the session that found the problem, so I did not edit the numbers. Silently changing a committed output I cannot re-run is exactly what the oracle rule exists to prevent. Instead that day's commit rewrote the debrief and teacher notes around the failure: what the plot claims, what the printed number says, why, and why Session 2 succeeds with the same recipe on the well-separated top of the spectrum.

Four days later, with a pinned environment available, the fix went in. The cell now runs the identical three-step recipe on two matrices. The original Wishart matrix still stalls at 0.5015 against 0.0097, kept as evidence. A second matrix, built with one eigenvalue deliberately isolated so the relative gap is about 0.21, converges to machine precision by roughly iteration 70: Rayleigh quotient and true minimum both print 1.00000000, and eigenvector alignment against `eigh` prints 1.000000. The new matrix draws from its own random stream so Session 2 is untouched; re-executing both sessions returned its outputs byte-identical, drift 2.22e-16 included.

Riemannian optimization guarantees you never leave the manifold, and it did that perfectly in both runs. It inherits ordinary gradient descent's conditioning problems exactly. **Riemannian optimization fixes the constraint, not the conditioning.** The teacher notes ask the room to predict which curve will look like a working algorithm before running the cell, because most students guess wrong: "Riemannian" sounds like the fix when the fix is the spectral gap.

## The narration pass that caught it

The reason I was reading outputs closely at all was a five-day pass from 2026-07-24 to 2026-07-28. Before it the repository had session markers and intuition cells everywhere, but zero teacher notes in any notebook, and only about one in five result-producing code cells was followed by any interpretation of its output, against the style guide's own rule.

The pass added two conventions and tooling to enforce them. Every result-producing cell gets a debrief that names the actual number, says what it proves, and says where it breaks down. Every session gets a collapsed teacher-notes block: timing, what to draw on the board first, the misconception students arrive with, a question to hand back to the room, and what to do if the demo misbehaves. The validator reports coverage for both, and a sync script re-derives the fill-in-the-blank student copies from their parents so prose cannot drift. Across the 84 commits of the pass, teacher notes went from 0 to 258 of 258 sessions and debriefs to 300 of 300 result cells, which is what the validator reports today.

Reading every output back to the reader means you cannot skip the ones that disagree with their titles. That is how narrating a failure became a habit.

## What the rule cannot do

Seventeen of the 87 workshops carry a draft banner because I could not execute them. They need things I did not have when I wrote them: MATLAB, an FPGA toolchain, a GPU, an RTL-SDR dongle, a microcontroller board, Docker, or an expert proof-read of the proofs. An oracle check only means something once it has been run, so those 17 are labelled unverified rather than listed quietly beside the 70 that executed clean under the pinned requirements.

That is the whole discipline. Every number gets a check the author did not derive; every check is read back to the reader; every failed check is narrated before it is fixed; and everything that could not be checked says so.
