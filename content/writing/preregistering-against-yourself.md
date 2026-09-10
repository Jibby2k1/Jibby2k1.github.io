---
title: Preregistering against yourself: six phases of trying to falsify a memory architecture
slug: preregistering-against-yourself
description: A research log on running a solo architecture idea like a clinical trial, and what it is like when six phases of honest testing return "no advantage established yet."
date: 2026-09-08
kind: research-journal
label: Research log
topics:
  - kernel methods
  - preregistration
  - sequence modeling
  - research methodology
  - hpc
heroImage: assets/img/blog/preregistering-against-yourself.svg
heroAlt: Timeline of six research phases, each with a locked decision memo and a validity gate, with the hypothesis narrowing from dense radial attention to sparse memory.
summary: Locked memos, validity gates, and paired-seed statistics turned a memory-architecture idea into a program where a negative result is as usable as a positive one.
relatedProjects:
  - kernel-adaptive-memory
takeaways:
  - Writing the decision rules before looking at results is the only reliable defense against my own optimism.
  - A validity gate that blocks interpretation is worth more than a significance test, because most of what it caught was not science but plumbing.
  - Six phases produced no general advantage, but one surviving signal narrowed the hypothesis into something sharper and cheaper to test.
---
Kernel Adaptive Memory (KAM) started as a bridge between two things I like: kernel least-mean-squares filters, which predict with localized basis functions and adapt cheaply online, and Transformers, which learn context-dependent routing. The first version replaced dot-product attention with a radial kernel score under a learned metric, added a bank of persistent support vectors, and kept a normalized least-mean-squares (NLMS) readout for adaptation after a distribution shift.

I liked the idea enough to distrust myself with it. So I ran it the way a clinical trial is run: every phase got a written brief, a set of validity checks that had to pass before anyone was allowed to interpret anything, and a decision memo mapping each possible outcome to an action. The [repository is public](https://github.com/Jibby2k1/KAM), and by August it held 34 commits, 19 briefs and memos for the phases, and 82 report files. Most of them say some version of "not established."

## Phase I: the grid I was sure would work

The first matched grid was 45 runs: five model variants, three tasks, three seeds. Dot-product attention won or tied on the copy task and the hidden-regime grammar. A dot-product hybrid won on Mackey-Glass, a delayed chaotic time series. On character-level Tiny Shakespeare, the dot-product Transformer reached perplexity 6.96 to KAM's 7.41, in about half the runtime. Copy length-generalization failed outright.

One thing survived. After a shift in the Mackey-Glass delay parameter, freezing the feature geometry and letting only the readout adapt with NLMS cut post-shift error by 44.5 percent against a static readout. I wrote that down and moved on.

## Phases II through V: hardening the method, not the model

Phase II was a five-seed paired grid, 50 runs, with bootstrap confidence intervals and Holm correction for multiple comparisons. No corrected comparison excluded zero. The decision memo has five questions and five answers: not established, not established, not established, not yet, none.

Phase III built a 1,728-row development manifest for the university cluster; nothing confirmatory was submitted. Phase IV was a bounded data-regime screen whose decision line reads "retain as diagnostic only"; no architecture was promoted from it. Phase V introduced the rule I now consider most important: a validity gate of structural checks that must pass before anything downstream runs or gets interpreted. Its pilot passed 144 of 144 rows; its larger sub-studies then blocked themselves on stream-generation and capacity-consistency checks, which is exactly what the gate was for.

## Phase VI: abandoning the original claim

The Phase VI brief opens with a sentence I had to write myself: "The broad claim that the current dense KAM bank should replace Transformer attention is not supported." The hypothesis narrowed. Instead of replacing attention, a standard decoder block gets a memory branch that routes each token to only the top-K of M supports by kernel distance, applies a small local expert at each, and re-enters the residual stream through a zero-initialized gate. "Separable" means two parameter classes on two timescales: fast algebra (the expert values, which admit closed-form or recursive solves) and slow geometry (the keys, optionally frozen late in training). That is the adaptation signal from Phase I, rebuilt as a component instead of a replacement.

An overnight pilot on four GPUs looked striking: fixed-key KAM reached validation loss 2.0081 against 2.8833 for a widened Transformer, 30.4 percent lower. With three paired seeds, the permutation p-value was 0.25 and the Holm-adjusted value 0.75. The locked rule filed it as "retain as diagnostic only," which the correction memo glosses as "run a clean, adequately powered confirmation before acceleration," not "KAM failed."

So I preregistered the confirmation: 156 rows at 50 million tokens each, 30 paired seeds on the primary corpus, a 2 percent minimum relevant improvement, and roughly 90 percent power at a planning effect deliberately far below the pilot effect, to protect against winner's curse. No optional stopping, no seed replacement, no peeking. Later documents describe the primary superiority gate as failed. The numeric report is not committed, so I will not quote it here. The honest summary is that no general advantage has been established.

## When the gate fails for a reason that is not science

The current thread is a mechanism study, a "behavioral atlas" of what sparse memory actually learns. Its 168-row, 50-million-token Stage 1 finished training and then failed its own preregistered validity gate: two rows failed a strict FP32 check that a matched permutation of keys and experts must leave predictions unchanged, and 118 failed the looser BF16 version.

That check exists because a memory bank whose output depends on the storage order of its supports is not measuring memory. The locked audit plan had four possible outcomes and forbade any efficacy interpretation. Replaying saved checkpoints localized the failure to late memory layers, reproduced it on three GPU hosts, and found it absent on CPU. The cause was CUDA top-k tie ordering: when routing scores tie at the K boundary, the GPU's selection order is not invariant to a permutation of the inputs. The fix is an opt-in tie-breaker keyed to a support identity that travels with the permutation. It passed a 26-checkpoint replay and a 24-row training pilot, and a fresh 168-row run was preregistered on August 9 with one declared difference from the original. The original campaign is not rehabilitated; it is shown only as historical context.

## What the audits actually caught

Independent audits of each phase were where the value of the process showed. None of these would have shown up as a bad loss, and every one would have produced a confident, wrong table.

- A plural-to-singular field mapping turned "geometries" into "geometrie," so every row silently ran the default geometry.
- Rows labelled with one retrieval task ran a fallback task instead.
- Optimizer labels were recorded, but a single joint loop ran regardless.
- Adaptation rows declared recursive least squares while the runner used full-model AdamW.
- A first full deployment was cancelled after 1,315 partial rows when the audit found that short schedules had produced zero geometry updates.

## What it feels like

Mostly it feels slow. Occasionally it feels like arguing with a version of myself who wants the 30 percent number to be real. The memos are how I win that argument in advance. What I have after six phases is a narrower, testable hypothesis about sparse, top-K-routed separable memory, a gate that catches tie-ordering bugs before they become results, and a program in which a negative answer is as usable as a positive one. That was the design goal, and it is the one thing I can say worked.
