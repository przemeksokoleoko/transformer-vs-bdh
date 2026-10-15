# Transformer vs BDH-GPU — Toy Memory Experiment

This repository contains a compact, self-contained toy experiment (transformer-vs-bdh.py) that compares a small Transformer to a BDH-style recurrent memory architecture on a synthetic associative recall task.

## Executive summary

- Task: Multi-Query Associative Recall (MQAR). Each sequence contains a short key→value table in the prefix (k1 v1 k2 v2 ... kN vN) followed by repeated queries that ask the model to output the previously paired values for shown keys. This isolates in-context associative memory and retrieval.
- Models compared:
  - Transformer: a small, 2-layer Transformer using RoPE and standard softmax attention.
  - BDH-GPU (BDH): a compact, recurrent, neuron-style memory architecture that uses ReLU activations, reuse of weights across passes, and a constant-size recurrent state.
- What the script measures: training loss, accuracy versus number of key-value pairs, activation sparsity, attention/retrieval mass focused on the correct value token, and a check that the parallel and recurrent BDH formulations produce the same outputs.
- Purpose: this is a mechanistic benchmark intended to probe how a sparse, recurrent memory-like architecture compares to transformer-style attention on a narrowly-defined associative-memory task. It is not meant to be a broad claim about general superiority.

## What the experiment tests (brief)

- Whether each model can store and recall key→value associations that appear only in context (in-context memory).
- How well the models generalize to longer contexts (by increasing the number of pairs at evaluation beyond the training distribution).
- Internal mechanisms used for retrieval: attention mass concentrated on correct value tokens and activation sparsity patterns (dense signed residual stream for Transformer vs. positive/sparse neuron activations for BDH).
- Memory/state cost comparisons: the script computes the number of floats needed to represent the Transformer KV cache versus the BDH recurrent state.

## Quick usage

Run the experiment directly:

```bash
python transformer-vs-bdh.py --steps 3000 --lr 1e-3 --bdh_layers 2
```

Flags:
- --steps: number of training steps (default 3000)
- --lr: learning rate (default 1e-3)
- --bdh_layers: number of BDH passes / layers
- --plot: save training curves and attention/activation figures
- --seed: random seed

## Summary critique (short)

This script is a well-motivated toy mechanistic probe but has limitations that prevent it from being a robust architecture benchmark by itself. Key critique points:

1. Single-task, single-setup: MQAR is a narrow synthetic task. Success here is informative about associative memory mechanisms but does not imply generalization to realistic sequence modeling problems.
2. Not fully apples-to-apples: the Transformer and BDH operate under different compute and memory models (per-token KV cache vs. recurrent state). The script reports parameter counts but does not explicitly match compute (FLOPs), latency, or memory-per-step.
3. Low statistical rigor: training is run once per model/seed in the default script. There are no repeated seeds, confidence intervals, or systematic hyperparameter sweeps.
4. Small scale: models and training budgets are small. While appropriate for mechanistic interpretability, they limit conclusions about scaling behavior.
5. Simplified evaluation: training and evaluation sample from the same random task distribution. There is no held-out validation set or separate out-of-distribution (longer sequences / distractors) test battery.
6. Limited baselines and ablations: alternatives (linearized attention, RNNs, Hopfield networks, other sparse-memory designs) are not included, and ablation experiments (remove RoPE, change Nh, remove gating) are not present.

## Checklist — how to improve the experiment and the implementation

A practical checklist you can follow to raise the experiment to a more reproducible, publication-quality benchmark.

1. Reproducibility and statistics
   - [ ] Run all experiments across multiple random seeds (e.g., 10–30) and report mean ± std.
   - [ ] Save model checkpoints and evaluation logs for each seed.
   - [ ] Add a minimal config file or CLI flags to fully specify hyperparameters, seeds, and training schedules.

2. Evaluation / datasets
   - [ ] Create explicit train / validation / test splits (different RNG seeds or held-out keys/patterns).
   - [ ] Add out-of-distribution tests: longer contexts (extrapolation), added distractor tokens, noisy or partial keys, repeated keys, and different key/value vocab sizes.
   - [ ] Measure wall-clock time, per-step latency, and peak memory usage (e.g., with torch.cuda.max_memory_allocated if using GPU).
   - [ ] Report FLOPs or an empirical proxy (tokens×model size×layers) to compare compute budgets fairly.

3. Fair comparisons and baselines
   - [ ] Match compute budget across models (e.g., match training FLOPs or wall-clock compute) in addition to reporting parameter counts.
   - [ ] Add baseline architectures: LSTM/GRU, linear attention variants (Performer, Linformer), Hopfield nets, and other memory-augmented RNNs.
   - [ ] Try variants of BDH: different Nh (neurons), H (heads), and L (passes) to see sensitivity.

4. Ablations and mechanistic checks
   - [ ] Ablate RoPE to see how positional encoding affects retrieval.
   - [ ] Ablate gating terms and the ReLU nonlinearity to measure their contributions.
   - [ ] Compare parallel BDH vs. recurrent BDH across different sequence lengths and show how state size scales.
   - [ ] Visualize failure modes (confusion matrices, attention maps when retrieval mass is low).

5. Metrics and plots
   - [ ] Add error bars to accuracy vs. pairs plots and show training curves for all seeds.
   - [ ] Add precision/recall if you convert the problem to a multi-class retrieval metric or if partial matches are possible.
   - [ ] Log/store per-layer attention and activation distributions; provide summary visualizations.

6. Implementation & code hygiene
   - [ ] Add unit tests for make_batch and for equivalence between parallel/recurrent BDH on small cases.
   - [ ] Factor experiment harness: separate model definitions, data generation, training loop, and evaluation into modules to enable easier experimentation.
   - [ ] Add type hints and docstrings for public functions.
   - [ ] Use deterministic device settings and document hardware used for experiments.

7. Reporting and claims
   - [ ] Limit claims to the scope of MQAR. State clearly in any write-up that results are limited to associative recall tasks under the evaluated regime.
   - [ ] If making broader claims, test on multiple tasks and architectures and provide rigorous statistical evidence.

## Next steps (suggested immediate changes)

1. Add a small driver script or a unit-test-like harness that runs the same experiment over 10 seeds and produces aggregated plots.
2. Add explicit logging to a CSV or JSONL file (step, seed, loss, acc, runtime). This enables plotting and comparison externally.
3. Add a short README (this file) that documents how to run and how to reproduce the main figures.
