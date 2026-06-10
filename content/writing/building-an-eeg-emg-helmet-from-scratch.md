---
title: Building an EEG/EMG helmet from scratch
slug: building-an-eeg-emg-helmet-from-scratch
description: Lessons from turning a modular acquisition concept into a real device that must deal with cables, noise, and human constraints.
date: 2025-10-01
kind: research-journal
label: Hardware
topics:
  - eeg
  - emg
  - hardware
  - neuroengineering
heroImage: assets/img/projects/Ergo.webp
heroAlt: Visual representing the Ergo biosignal acquisition project.
summary: Biosignal research gets cleaner when hardware, firmware, and modeling are treated as one system instead of three separate concerns.
relatedProjects:
  - ergo
takeaways:
  - Signal quality is a systems problem, not only a modeling problem.
  - Human factors and hardware constraints shape what research is actually feasible.
  - A repeatable acquisition stack is a prerequisite for trustworthy neuroengineering work.
---
The interesting part of a biosignal system is rarely only the final classifier. The real work starts earlier, at the layer where electrode contact, cable routing, grounding, shielding, synchronization, and physical comfort decide whether the data is usable at all.

Working on an EEG/EMG helmet makes that obvious very quickly. Small mechanical choices change the noise floor. Convenience features become signal-quality decisions. A design that looks elegant in a block diagram can become unmaintainable the moment a person has to wear it for a full session.

That is why I treat hardware, firmware, and analysis as one system. Acquisition quality constrains the models you can train, the features you can trust, and the experiments you can repeat. A poor interface upstream creates ambiguity all the way down the pipeline.

The payoff is that once the physical system is disciplined, the modeling questions become cleaner. You can spend more time on dynamics, stability, and interpretation, and less time guessing whether a result is just a wiring artifact.
