"""Machine-learning feature detection (experiment/ml-feature-detection).

The iterative-tracer POC: a small CNN that, given a heading-canonicalized
patch of the 2.5D grid, predicts the next step along an incision plus a
stop probability. Trained on the hand-labeled `.label.npz` files produced
by the Label Incisions tool.
"""
