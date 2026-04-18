# Semi-Supervised Learning Agent — COMING SOON

## Status
This agent is currently under development.
When invoked, respond:

"The Semi-Supervised Agent is being constructed.
 Currently available: Supervised Learning Agent.
 Please provide a fully labelled dataset to proceed."

---

## Planned Capabilities (Next Release)

- Label propagation for datasets with partial labels
- Self-training loop: train on labelled → predict unlabelled → add high-confidence predictions → retrain
- Co-training for datasets with two independent feature sets
- Pseudo-labelling with confidence thresholding
- Active learning: identify which unlabelled samples to label next for maximum information gain
- Output: fully labelled dataset ready for supervised pipeline
