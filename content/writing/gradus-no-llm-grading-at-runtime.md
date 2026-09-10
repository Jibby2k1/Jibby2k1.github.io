---
title: Why Gradus refuses to let a language model grade anything at runtime
slug: gradus-no-llm-grading-at-runtime
description: The March 2026 decision to keep language models out of the Gradus grading loop, what deterministic diagnosis actually requires, and what the boundary costs.
date: 2026-09-08
kind: research-journal
label: Design log
topics:
  - adaptive learning
  - deterministic grading
  - product design
  - education
heroImage: assets/img/blog/gradus-no-llm-grading-at-runtime.svg
heroAlt: Diagram of a content factory feeding course packages through a validation gate into a runtime engine marked no language model, where grading and diagnosis happen.
summary: Gradus grades and diagnoses with rules, not a model, so every outcome is reproducible; the price is a heavy authoring contract and an audit for hollow content.
relatedProjects:
  - gradus
takeaways:
  - A deterministic grader makes every learner outcome reproducible and A/B testable, which is the property a research-style pilot needs most.
  - Diagnosis without a model is a lookup over author-declared fail points, so the authoring contract carries the intelligence.
  - The boundary between content generation and runtime grading is the product; validation gates and a substance audit are what keep it honest.
---
The landing page for Gradus shows a stat that reads "0 runtime LLM grading decisions". This post is about why that number is zero, what it takes to keep it there, and what it has cost me.

## The decision

Gradus began on 2026-03-07, under the working name Piso, as an adaptive engine for a proof-based real analysis pilot. The first architecture decision record (ADR), written the same day, settled the grading question before any product existed: the V1 runtime uses deterministic grading and diagnosis logic, and language models may be used only offline for content drafting.

The reasoning in the ADR is short. A pilot has to be explainable, stable, and measurable if I ever want to run research-style experiments on it. Grading with a large language model (LLM) at runtime introduces nondeterminism and operational risk. Two alternatives were considered and rejected: fully LLM-driven grading, and a hybrid where a deterministic grader falls back to a model for anything it cannot handle. The listed benefits are reproducible outcomes, easier A/B testing, and simpler debugging. The listed cost is reduced flexibility for free-form responses. The ADR leaves a follow-up for a controlled natural-language evaluation path, which is still open.

Two months later the positioning docs turned that constraint into the product wedge: Gradus is not an answer machine, and it should not rely on unverified runtime LLM grading for learner-facing correctness decisions.

## What deterministic grading actually requires

Rule-based grading sounds simple until a learner types LaTeX. The item family schema allows five answer modes: choice, exact, regex, sequential, and latex_exact. The interesting work is in normalization, which was added the day after the ADR.

For latex_exact the normalizer lowercases, strips `$`, `$$`, `\(...\)`, and `\[...\]` delimiters, maps Unicode inequality symbols to their macros, removes `\left` and `\right`, applies built-in aliases such as `\le` to `\leq` and `\to` to `\rightarrow`, applies per-concept aliases that the author declares, and finally collapses whitespace around braces, brackets, and operators. Only then are learner and canonical strings compared. Sequential mode grades a fill-in-the-blank chain step by step, each step with its own mode, and scores partial credit as matched steps over total steps. Choice mode compares the selected set to the canonical set exactly.

Diagnosis is where a model would be most tempting, and it is where the rule-based design is most exposed. The error taxonomy has twelve values including "none": eight universal types (missing prerequisite, recall miss, reasoning miss, confusion miss, missing step, wrong application, scope error, careless miss) and three proof-math extensions (quantifier error, theorem-condition error, counterexample gap). On a wrong choice, the engine looks up the distractor's declared fail point and reads the error type from it. If no mapping exists, it falls back to the concept's first author-defined fail point, and failing that to a stage heuristic: recognition and recall stages default to a recall miss, structured-completion and reasoning stages to a reasoning miss. Repeated misses on the same concept escalate to a prerequisite miss.

The engine records a confidence proxy of 0.35 for any miss and 0.55 when repeated misses trigger a prerequisite backtrack, so the learner-facing copy calls every miss diagnosis a best guess. Whether the fail point came from a distractor mapping or a fallback is recorded separately, in the ranked-weakness list, not in that number.

Each error type then routes to a repair target. Recall and careless misses stay on the same concept. Reasoning, missing-step, quantifier, theorem-condition, and counterexample errors run a backward breadth-first search over the prerequisite graph and put weak prerequisites first. Confusion, wrong-application, and scope errors look at declared neighbor concepts, sorted by mastery, before the traversal. A prerequisite miss traverses prerequisites and neighbors together. Ties are broken by graph distance, then mastery, then concept order, so the same learner history always produces the same next question.

## Where language models are allowed

The zero on the landing page is not a claim that Gradus avoids language models. It uses them heavily, but outside the grading loop and behind validation gates. Course packages are generated by `tool/generate_course.dart`, which can run through a coding agent reading prompt files and writing JSON, or through an API. The app itself also has a course builder under its developer and authoring tools, so "offline" in the ADR's sense is no longer literally true; that builder runs the same validate-and-repair loop. Every package then passes `tool/validate_and_repair_course.dart`, which checks schemas and, with `--strict-latex`, promotes LaTeX warnings to blockers. Generated learning sidecars add reading text and audio scripts to a lesson, and narration is rendered from that validated text by a local text-to-speech model or an API voice; the audio itself is not separately inspected.

The boundary is the product. Every course package a model drafts passes validation before a learner sees it, and nothing a learner writes is judged by a model. That is what lets me say a wrong answer maps to the same fail point today, tomorrow, and in the control arm of an experiment.

## What it cost

The first cost is authoring burden. The runtime has no intelligence of its own, so every concept has to carry prerequisites, neighbors, author fail points with error types, distractors mapped to fail points, and LaTeX aliases. Any concept missing that contract degrades to the stage heuristic.

The second cost was subtler. Generated courses were passing the schema and LaTeX gates and still being hollow. In July I wrote a substance audit (still uncommitted as I write this) that normalizes each prompt, its choices, and its canonical answer into a skeleton, replacing the concept title with a topic placeholder and math with a math placeholder, and flags any skeleton shared by three or more families across three or more concepts, plus a list of known template frames. Calibrating it against the hand-authored analysis pilot gave roughly one percent boilerplate; template-stamped generated courses scored between 83 and 98 percent and failed the gate. Only the flagged families are regenerated, rather than the whole course.

That audit exists because of the ADR. When the grader is a rule, the content has to carry the meaning, and structural validity is not the same thing as substance.

## Where it stands

Gradus is in a closed learner beta with a [public demo](https://gradus.raulv.dev/demo) that needs no sign-in. No school pilot has run and there are no efficacy results; the deterministic design is what would make such a pilot measurable, not evidence that it works.
