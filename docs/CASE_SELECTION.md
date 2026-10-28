# Deterministic illustrative case selection

Cases are selected only after the relevant final evidence gate authorizes the
underlying comparison. Eligibility is defined by the registered case type, the
held-out test partition, and availability of every compared method on the same
example. Eligible rows are ranked by SHA-256 of the frozen namespace, model,
dataset, example identifier, and seed 2027; the first two are retained.

The procedure records the eligible population size and selection hashes. It
never examines partial RQ4/RQ5 outcomes, never substitutes a visually appealing
example, and supports illustration only—not confirmatory inference or effect
estimation.
