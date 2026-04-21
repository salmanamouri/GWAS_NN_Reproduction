# Repo audit

## Architecture
- Encoder:
  Sparse linear layer (SparseLinearLayer) mapping SNPs → gene layer using mask
  (NOT per-gene MLP as described in paper)

- Predictor:
  MLP: num_genes → 100 → 1

- Main effect model:
  Separate neural network (Main_effect) modeling only additive effects (no interactions)

- Activation:
  Softplus(beta=10) ❌ (paper uses ReLU)

- Hidden sizes:
  Predictor: [num_genes → 100 → 1]
  Encoder: linear only (no hidden layer)

---

## Training
- learning rate:
  0.001 ❌ (paper: 0.005)

- epochs:
  200000 (very large, not explicitly stated in paper)

- batch size:
  int(n_samples / 100) ❌ (paper: 30000)

- early stopping:
  not implemented

- regularization:
  no explicit L1 CV shown (paper: 5-fold CV)

---

## Interaction scoring
- function:
  GlobalSIS (Shapley Interaction Score approximation)

- baseline:
  Uses mean prediction as baseline

- sample selection:
  Top and bottom predicted samples (num_samples = 200)

- Monte Carlo / exact:
  Monte Carlo approximation (num_permutation = 10)

---

## Permutation
- Perm I:
  Implemented:
  y_null = predicted_main + permuted_residual

- Perm T:
  Not implemented

- Perm R:
  Not implemented

---

## Data assumptions
- genotype file:
  CSV, shape: (n_samples, n_snps)

- phenotype file:
  CSV (single column)

- snp size file:
  text file listing number of SNPs per gene

## What the public repo provides
- Sparse gene interaction NN reference implementation
- Toy CSV-based workflow
- Interaction runner
- Permutation runner

## What the public repo does not provide directly
- Full simulation pipeline for Figures 2–5
- Six baselines
- Full figure generation
- Deep ensemble orchestration
- 5-fold CV experiment harness

## Files to keep as references
- models.py
- functions.py
- run_interaction.py
- run_permutation.py

## Files to add for the reproduction
- repro/simulators/complex_simulator.py
- repro/simulators/null_simulator.py
- repro/simulators/simple_simulator.py
- repro/scripts/run_null_calibration.py
- repro/scripts/run_perm_benchmark.py
- repro/scripts/run_complex_benchmark.py
- repro/scripts/run_simple_benchmark.py


