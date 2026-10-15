---
title: A research loop across two clusters
slug: a-research-loop-across-two-clusters
description: How we organised hundreds of surrogate-model experiments across a university GPU cluster and a national-lab machine, with written targets, small screens before large grids, and gates that caught our own mistakes.
date: 2026-10-15
kind: research-journal
label: Research log
topics:
  - research methodology
  - hpc
  - scientific machine learning
  - reproducibility
heroImage: assets/img/research/cfd/capacity-grid.webp
heroAlt: Heat map of median accuracy over network depth and width, one of the large grids run on the national-lab machine.
cover: generated
summary: Written targets, a promotion rule from small screens to large grids, bit-for-bit gates and canaries let a fast experiment loop stay honest; most of what the gates caught was plumbing, and one catch was a conclusion we had drawn too early.
relatedProjects:
  - cfd-surrogate-modeling
takeaways:
  - Write the targets down before the experiments, including what would count as a failure, so results are judged against a bar rather than against hope.
  - Screen ideas small and cheap, and promote only what passes a stated bar to the expensive grid; the grid then answers several factors at once.
  - A canary that reproduces a known result bit for bit is the cheapest experiment you will ever run, and it is where most of the mistakes surface.
---
The results in the [shared-surrogate](inputs-before-capacity.html) and [step-size](when-converging-is-the-wrong-goal.html) posts came from several hundred training runs over a few days. Running that many experiments quickly is easy to get wrong in ways that never show up as a bad loss curve. This post is about the structure that kept the work honest, because the structure is more reusable than any single result.

## Targets first

Before the architecture and learning-system work started, we wrote down four targets, each with a number that would count as done:

- **One model as good as many.** A model shared across all 36 cases should come within 0.01 R² of per-case models.
- **Stable refinement.** The self-consistent state should never end more than 0.01 below the fixed state, on any case, at 32 outer cycles.
- **Transfer.** On a held-out split, refinement should improve every unseen case.
- **Constraints.** Keep exact rotation equivariance and the paper's evaluation, and leave one snapshot untouched for a single final confirmation.

The point of a written bar is that it makes an ambiguous result look ambiguous. When we later found that the "one model as good as many" comparison had used the wrong baseline, the target made the correction mechanical: the number moved from just above the bar to below it, and the claim was withdrawn in the log.

## Two clusters, two jobs

The work ran on two machines with different strengths. The university cluster (HiPerGator, NVIDIA L4 GPUs, a shared allocation) handled small screens: one idea, eight cases, two to four labels, a few hours. The national-lab machine (LLNL Tuolumne, AMD MI300A APUs) handled everything that only means something at scale: shared models over all cases, held-out splits, factorial grids, many seeds, long runs. At peak we had around 45 of its nodes busy.

An idea moved from the first to the second only if it passed a stated bar on the screen: at least +0.005 median R² over the reference on the same labels, and no new collapse. Ideas that only make sense at scale, like the capacity of a shared model, went straight to the large machine after a canary. Large runs were designed as grids rather than one factor at a time. A single submission compared raw and log-scaled inputs, two architectures, three refinement rules and two evaluation protocols at four seeds, and answered several questions at once.

Porting was its own small project. The equivariant layers rely on batched matrix products, and on the MI300A one library setting made them about 260 times slower than necessary; turning it off brought throughput in line. We then required agreement with the L4 results within seed-to-seed scatter before trusting any number from the new hardware, rather than bit identity, which different hardware cannot give.

## Gates that reproduce a known answer

Every wrapper around the original training code, whether it widened the case tables, changed the step rule, swapped the architecture, or shifted the reference frame, had to reproduce a stored result bit for bit with the new option switched off before any new number was read. Every new job type started with a canary, a two-minute run on three cases that had to finish and pass a specific check.

What these caught was mostly plumbing, and every item would have produced a confident, wrong table:

- A canary's acceptance threshold was tighter than its own Monte Carlo noise. The code was right and the check was wrong; we replaced it with a test against the standard error.
- A shell idiom that is harmless almost everywhere (`[ a != b ] && …` inside a pipeline with `pipefail`) silently killed three of four leave-one-out folds on one node, precisely the folds the experiment was about.
- A dropout variant collapsed at evaluation. Tracing it showed the problem was where the dropout sat (on equivariant feature channels), not dropout itself; two other placements were safe.

## Negative results are results

A good fraction of the conclusions are negative: adaptive step rules do not help during training, wider and deeper networks do not help once the inputs are scaled, a label-free safeguard does not make extrapolation safe, and converging the state fully at inference makes predictions worse. Each of these was a reasonable idea, each was tested at a size that could have shown an effect, and each is recorded with the same care as the positives. Negative results are the cheapest way to narrow a research program, and they are the results most likely to be quietly dropped if the process does not insist on keeping them.
