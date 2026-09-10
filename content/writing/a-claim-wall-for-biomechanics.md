---
title: A claim wall for biomechanics
slug: a-claim-wall-for-biomechanics
description: How the Ora mechanics engine encodes what a result may and may not say as two enums, and why crossing that line is a reviewed decision rather than a code change.
date: 2026-09-08
kind: research-journal
label: Research log
topics:
  - biomechanics
  - software design
  - verification
  - digital twins
heroImage: assets/img/blog/a-claim-wall-for-biomechanics.svg
heroAlt: Diagram of a brick wall separating allowed mechanics outputs from barred outputs, with a single gated door labeled reviewed decision.
summary: A mechanics engine for lifters is only trustworthy if the outputs it cannot support are unreachable by construction, not merely omitted from the current release.
relatedProjects:
  - ora
takeaways:
  - Encoding forbidden outputs as a typed enum that every packet must list makes the boundary a compile-time fact instead of a documentation promise.
  - Moving a quantity across the wall is a reviewed decision with a named validation bar, and an audit showed the nine barred outputs need at least three different bars.
  - A conservation check that only tests where a term vanishes cannot see that term; a sub-interval work-energy check caught a bug the endpoint check had passed.
---
The mechanics engine inside Ora, the training app I co-founded with a small student team, answers one narrow question: given two exercise setups, how does the joint moment differ across the range of motion? It is a deterministic, planar, quasi-static solver: not a muscle model, not an injury model, not a clinician. The design problem I care about most is making that sentence true in code, not only in a README.

## Two enums, one wall

What a comparison packet may say is encoded in two enumerations in the contracts module. `AllowedMechanicsOutputV1` has nine members: modeled point positions, external forces, moment arms, external and balancing joint moments, a balancing-moment summary, the difference between two setups, finite-difference sensitivities, and bounded uncertainty. `UnavailableMechanicsOutputV1` has nine members too: muscle activation, individual muscle force, joint contact force, tendon force, ligament stress, tissue stress, injury probability, medical interpretation, and three-dimensional dynamics.

The second list is not a comment. Every packet carries an `unavailable_outputs` field, and the validator rejects any packet whose set does not equal the full required tuple. A packet cannot quietly drop "injury probability" from the list of things it refuses to say, and there is no field where a muscle force could go, so the Dart app that reads the released response surface has nothing to read.

I call this the claim wall, and the project's working agreement names it the principal asset.

## Promotion is a decision, not a diff

The rule that goes with the enums is short. Promoting anything from unavailable to allowed is a separate, explicit, reviewed decision with a named validation bar, never a side effect of another change. A named bar means writing down what the output is checked against, how closely it must agree, and who reviewed that.

Once the decision to pursue muscle-level outputs had been taken, and before anyone set a bar, I audited what promoting each of the nine would cost. The flat enum hides that they are three different kinds of barred.

- One is barred only because it is unfinished as a packet output. Three-dimensional dynamics is missing the time dimension, not a concept, and its gate is a reduction test we already know how to write: a dynamic solve must reproduce the quasi-static result exactly at zero acceleration.
- Four are model-dependent. More muscles cross a joint than there are equilibrium equations, so a muscle force is a consequence of a cost function, not a measurement. Checking one ranges from a consumer surface EMG sensor to an instrumented surgical implant.
- Two are not mechanics claims at all. Injury probability is epidemiology; medical interpretation is a regulated act.

The remaining two, ligament stress and tissue stress, sit between the second and third kinds, model-dependent to compute and clinical in use, and the audit left them unresolved.

The audit set no bar. It filled in the "against what measurement" column, proposed an agreement figure only for the reduction test, and left "who reviewed" blank in every row, because that is a human decision. The proposal that came out of it was a research tier for the model-dependent quantities, barred from user-facing packets by the same mechanism, with the assumption behind each number carried as a required field.

## The wall does not make the allowed side correct

A boundary on what you may say does nothing for the accuracy of what you do say. One bug in August is my clearest example of an oracle passing while wrong.

The task runner supplied the implement, a dumbbell in three of the four shipped movement tasks, as a point force at the distal end, which contributes to the external load and nothing else. So the dumbbell had weight but no inertia, and since it outweighed the limb segments, most of the moving mass accelerated for free. The kernel's own docstring warned that a body with weight has inertia. The runner ignored it.

The work-energy check at the time compared net joint work to the change in kinetic plus potential energy over the whole movement. A movement that starts and ends at rest has zero kinetic energy at both endpoints, so the missing inertia contributed nothing there and the check passed. Over sub-intervals, the gap at mid-movement was the dumbbell's kinetic energy to three figures.

The fix was four lines. Landing it moved most of the dynamic figures the project had recorded: every peak moment fell, because the implement's inertia now helps at the top of the range where the movement decelerates. Two comparative headlines reversed direction, one claim split into a true half and a false half, and the findings that survived best were stated as ratios rather than magnitudes. A second copy of the defect turned up in a test fixture that never called the runner, found because one figure did not move when everything comparable did. The sub-interval check is now permanent in the suite.

The lesson I wrote down: ask of any conservation check whether it vanishes exactly where the thing you are checking vanishes.

## Carrying the discipline into the twin

Since mid-August my time has gone into canonical-msk-twin, a pure-standard-library finite-element muscle-tendon-bone research demonstrator beside the mechanics engine. It is headed toward a human-scale frame with a viewer; its honesty statement calls every result a structural benchmark: not subject-specific, not biologically validated, no medical interpretation.

The same wall is there, expressed differently. Validation status is a field with two values, and the validated one is accepted only when a covering validation record is attached; no result so far carries one. A claim guard scans every claim-bearing text field for "validated", "subject-specific", "injury risk", and "diagnosis", and raises without such a record. It understands negation, so "unvalidated" passes and "validated" alone does not.

A design contract, at section 82 as I write this, pins interfaces before any increment is implemented; a decision log with 97 entries records each choice and reverses one only by adding a new entry. The loop protocol that drives the work lists refusals it must make: never promote anything out of unvalidated, never soften a claim to make a milestone reachable, never fabricate a mechanism to fill a gap. Scaling the model up buys geometry and composition, never validity.

## What I would keep

The enums cost almost nothing to write and have paid for themselves by turning "should we say this?" from a judgment call at review time into a question with a documented answer. The harder lesson is that the allowed side needs oracles that test where the terms do not vanish. I intend to keep both habits.
