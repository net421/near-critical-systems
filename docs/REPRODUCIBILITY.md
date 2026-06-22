# Reproducibility protocol

## 1. Create an isolated environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

## 2. Run tests

```bash
python -m pytest tests -q
python scripts/smoke_test.py
```

## 3. Verify the archived submission outputs

```bash
python scripts/verify_definitive_results.py
```

This verifies cardinalities and the numerical claims used in the manuscript.

## 4. Reproduce deterministic analyses

```bash
python scripts/run_all_paper.py --skip-beta --no-plots --out reproduced
```

This regenerates the MDP, hazard, capacity, cost-ratio, robustness, and lemma
analyses without rerunning the expensive Monte Carlo calibration.

## 5. Reproduce the definitive beta calibration

```bash
python -u scripts/run_beta_definitive_jupyter.py --threads 16 --save
```

The defaults are 15,000 replications, 300,000 periods, seed `20260429`, and 42
canonical validity-domain points. The script prints progress and an ETA. With
`--save`, it checkpoints after each point and can resume after interruption.

The beta runner generates the original 86 candidate-grid seeds before
deduplicating and selecting the 42 fitted points. This preserves the seed
mapping of the full paper pipeline while avoiding simulations that do not
enter the fitted model.

## Random-number generation

Configuration and trajectory seeds are generated hierarchically using NumPy
`SeedSequence.spawn`. Every Numba-parallel trajectory receives a distinct
child seed. Do not use the historical `--legacy-beta-seeds` option for the
submitted analysis.

## Expected environment

The definitive archived run used Python 3.12.13, NumPy 2.4.6, pandas 2.3.3,
Linux x86-64, and four Numba threads. The same output was independently matched
on Windows with Python 3.14.3, NumPy 2.4.3, SciPy 1.17.1, and Numba 0.65.0.
