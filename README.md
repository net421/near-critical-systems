# Reliability-Based Threshold Control for Near-Critical Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20792811.svg)](https://doi.org/10.5281/zenodo.20792811)

This repository contains the definitive research code, results, theory notes, and manuscript/submission materials for **Paper A**:

> *Reliability-Based Threshold Control for Near-Critical Capacity-Constrained Systems: Corrected First-Passage Dynamics and Hazard Characterization*

The repository is the foundational layer of the broader near-critical research program. It contains the theoretical model, corrected inverse-Gaussian calibration, hazard characterization, threshold-control experiments, tests, paper tables, and submission package.

**Archived software release:** [https://doi.org/10.5281/zenodo.20792811](https://doi.org/10.5281/zenodo.20792811)

## Quick access

- **Read the manuscript:** [`manuscript/submission_package/01_anonymized_manuscript_CAIE.docx`](manuscript/submission_package/01_anonymized_manuscript_CAIE.docx)
- **Submission package:** [`manuscript/submission_package/`](manuscript/submission_package/)
- **Theory notes:** [`docs/theory/`](docs/theory/)
- **Reproducibility guide:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- **Result-to-claim map:** [`docs/RESULTS.md`](docs/RESULTS.md)
- **Core library:** [`src/buffer_policy/`](src/buffer_policy/)
- **Reproduction scripts:** [`scripts/`](scripts/)
- **Archived results:** [`results/`](results/)

## Research ecosystem

```text
Near-Critical Systems / Paper A
        ↓
Controlled Near-Critical Benchmark / Paper B
        ↓
Supply Chain Digital Twin application
```

Related repositories:

- [Controlled Near-Critical Benchmark / Paper B](https://github.com/net421/controlled-near-critical): reproducible benchmark package connecting the theory to applied experimental validation.
- [Supply Chain Digital Twin](https://github.com/net421/Supply-Chain-Digital-Twin): industrial Digital Twin application extending near-critical logic into a networked supply-chain setting.

## Main model

The inventory recursion is

```text
I(t+1) = min{I(t) + Y(t) - D(t), S_max}
Y(t)   = C with probability 1-p, and 0 with probability p
D(t)   ~ Poisson(lambda)
tau    = inf{t >= 1 : I(t) <= 0}
```

The stability margin is:

```text
delta = (1-p)C - lambda > 0
```

Near-critical operation corresponds to small positive `delta`: the system is stable in expectation but remains vulnerable to collapse under disruption sequences.

## Main contribution

This work treats disruption-driven collapse as a first-passage reliability problem and develops a reliability-based threshold-control framework for near-critical capacity-constrained systems.

The paper:

1. characterizes first-passage collapse dynamics near the stability boundary;
2. shows that the classical inverse-Gaussian diffusion approximation systematically overstates first-passage variance in the near-critical regime;
3. introduces a corrected inverse-Gaussian shape parameter;
4. studies hazard structure and decreasing-failure-rate behavior through survivorship selection;
5. formulates an average-cost Markov decision process; and
6. computes reliability-based threshold policies that improve over age-based intervention.

## Definitive calibration

The submitted calibration uses:

- 15,000 replications per design point;
- a 300,000-period horizon;
- 42 unique points in `0.05 <= p <= 0.35` and `2 <= delta <= 15`;
- master seed `20260429`;
- hierarchical NumPy `SeedSequence` child streams; and
- Numba-parallel trajectory simulation.

The fitted correction is:

```text
beta(p, delta) = 0.247715 * p^(-(0.392927 + 0.0111267*delta))
```

with raw-beta `R^2 = 0.9536`. Using
`100*|eta_prediction-eta_fitted|/eta_fitted`, mean absolute error decreases
from `36.0%` to `5.0%`; maximum error decreases from `62.6%` to `18.4%`.

## Repository layout

```text
src/buffer_policy/                 Core library
scripts/                           Reproduction and validation entry points
tests/                             Unit and numerical-audit tests
results/beta_calibration/          Definitive corrected-RNG calibration
results/paper_tables/              Policy, hazard, robustness, and lemma tables
docs/                              Results, reproducibility, and theory notes
docs/theory/                       Theoretical framework notes
manuscript/submission_package/     Manuscript and editorial submission materials
```

## Paper and submission materials

The manuscript and submission documents are stored under:

```text
manuscript/submission_package/
```

This includes the anonymized manuscript, title page, highlights, cover letter, declaration of interest, data/code statement, editorial metadata, source audit, and validation report.

## Installation

Python 3.11 or newer is required. Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Verify archived results

```bash
python scripts/verify_definitive_results.py
python -m pytest tests -q
```

## Reproduce the beta calibration

For a Jupyter or many-core environment:

```bash
python -u scripts/run_beta_definitive_jupyter.py --threads 16 --save
```

Replace `16` with the available CPU-thread count. This is a CPU/Numba calculation; the current implementation does not use a GPU. The full run can take substantial time. The archived output is under `results/beta_calibration/`.

## Reproduce the remaining paper tables

```bash
python scripts/run_all_paper.py --skip-beta --no-plots --out reproduced
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete workflow and [docs/RESULTS.md](docs/RESULTS.md) for the result-to-claim map.

## Citation

If you use this repository, please cite the archived Paper A software release:

```text
Reliability-Based Threshold Control for Near-Critical Systems [Software and reproducibility package]. Zenodo. https://doi.org/10.5281/zenodo.20792811
```

## License

MIT License. See [LICENSE](LICENSE).
