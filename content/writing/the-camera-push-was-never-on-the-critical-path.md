---
title: The camera push was never on the critical path
slug: the-camera-push-was-never-on-the-critical-path
description: How the capture feedback loop in Media Cull Suite closed with hand-dialed recipes and a log before any camera code worked, and why the live push ended up narrower than "control the camera."
date: 2026-09-08
kind: research-journal
label: Design log
topics:
  - photography tooling
  - feedback loops
  - camera sdk
  - product design
  - local-first
heroImage: assets/img/blog/the-camera-push-was-never-on-the-critical-path.svg
heroAlt: Four loops labeled Capture, Cull, Deliver, and Memory; the Capture loop's backward feedback arrow from a longitudinal log to the next shoot is highlighted.
summary: The capture loop closed with hand-dialed recipes plus a longitudinal log; the Sony live push arrived last, set four of six fields, and was an optimization.
relatedProjects:
  - media-cull-suite
takeaways:
  - A feedback loop closes when you can measure the effect of an action, not when the action is automated.
  - Hardware control was narrower than expected; four of six recipe fields push live, and the mode dial decides whether anything is writable at all.
  - Freezing the node-graph editor followed from the same rule as the loop; new work goes into orchestration, not manual control.
---
Media Cull Suite is a local-first photo and video culling tool I built over a 26-day sprint this spring. Its distinctive idea is a capture loop: analyze the last shoot's failures, propose corrected camera settings for the next one, and measure whether the correction helped. The obvious way to make that feel real is to have the software talk to the camera. This is a log of why that was wrong.

## The forward half was done and the loop was still open

The first three capture versions landed on the same day in late May. `pca insights` clusters a shoot into situations by time of day, GPS, and CLIP scene embeddings when present, then emits a failure signature per situation: percentage underexposed, blurred, noisy, median ISO and shutter, and a plain-English diagnosis. `pca recipe` turns each signature into corrective settings for my body, a Sony α6700, with a one-line reason per setting. A `pca.camera` package scaffolded the live push: a planner, a mock controller for tests, and a real `SonySDKController` that was two stubbed methods waiting for a session with the camera plugged in.

Everything pointed forward. Nothing answered "did it work?" Keeper rate was computed per project. Two rainforest shoots a week apart had no shared identity. A recipe was fire-and-forget; nothing recorded that it was applied, so no outcome could be attributed to it.

The sentence in the gap analysis that reorganized the schedule: the data structures all point forward, and the demo needs an axis that points backward. The stubbed SDK methods were not the blocker. The missing log was.

## What a closed loop actually needs

The improvement curve needs three things: several shoots of the same situation type, a recipe applied before the later ones, and a store that links them and tracks a metric over time. None of that requires camera control. The photographer's fingers are a valid delivery channel. A recipe card dialed in by hand, plus a longitudinal log, is a complete feedback loop.

So the camera session was deprioritized and the backward axis got built. `pca.capture_log` is a JSON file holding one metrics snapshot per project and stable situation type, with an optional `applied_recipe` field. Situation types are proposed and always user-overridable; two shoots are never silently merged. `pca.trends` aggregates the log into a curve, and `pca trend` prints it as a sparkline.

The honesty rules were decided before the code. Keeper rate is contaminated by taste: a pickier cull lowers it regardless of what the camera did, so the curve always pairs it with the objective failure metric that was worst at the start. With one data point the view says so instead of drawing a trend. And the wording is association, not causation: "after applying this recipe, X moved," never "this recipe caused X."

Two more pieces landed before any camera was involved: a pre-shoot briefing that reads the log into tomorrow's recipe and renders an offline single-file HTML field card for a phone, and a push log so that a push recorded before a shoot becomes the `applied_recipe` stamp after the cull, with no manual flag. All of it ran end to end against the mock controller.

## The push arrived last, and it was smaller than "control the camera"

The live push worked eleven days after the scaffold, on 2026-06-09. Sony's Camera Remote SDK is large and license-restricted, so it stays out of the repo; a small C++ helper is built against it locally and the Python side talks to it over a JSON protocol.

- The SDK only talks to the camera in PC Remote mode. The α6700 also has a USB Streaming mode, and in that mode the SDK reports "no camera" even though `lsusb` sees the device.
- With the mode dial on Auto, the camera owns white balance, ISO, and exposure, and every recipe property is read-only; each write returns error 0x8402. The dial has to be on M before any push does anything.
- The Kelvin white-balance value is not remotely settable on this body. The mode switches to Color Temperature, but the temperature itself is dialed by hand.
- Minimum shutter for ISO Auto is a P/A-mode feature. In M you set the shutter directly, so the field does not apply.

The net is that four of the six recipe fields push live: creative look, metering, the ISO-auto ceiling, and exposure compensation (the last only when the physical EV dial is in a controllable position). The other two are reported as advisory, never as a hard error. And because the SDK cannot author the memory-recall dial slots, `--register MR1` after a live push records the slot and prints the one-time on-camera save, so in the field it is one turn of the dial.

"Control the camera" turned out to mean "set four settings, in one mode, with the dial in the right place." Useful, because the push stamps the log itself. But it is an optimization of a loop that was already closing in mock mode, and a bad place to have spent the first two weeks.

## The same rule froze the editor

The next day brought a decision that looks unrelated and is not. The suite has a React Flow node-graph editor with sixteen node types and per-node previews. The vision doc I wrote for the project says the product is the orchestration layer and the editor is the escape hatch: the user states intent, the system proposes a minimal, reversible action, a human approves. A node graph exposes the processing DAG directly; it works against that.

So the rest of the editor roadmap (history scrubber, direct-manipulation handles, conditional gating nodes) was dropped in the commit stamped v0.41, not deferred. The editor stays as the control surface for when you want control; new capability goes into loops. The same day, `pca tonight` landed: one command for the end of a shooting day that analyzes today's card, logs it, pushes tomorrow's recipe (mock by default, live if the camera is there), regenerates the field card, and prints the trip curve so far. Every step degrades to a note instead of aborting.

The through-line is the same in both decisions. A loop is closed when you can measure the effect of an action, not when the action is automated, and the surfaces worth building end in something a person wants: a curve that moved, a card on a phone, a dial turned to 1. The project is paused with all of that in place; the last commit is stamped v0.42.
