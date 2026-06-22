# Near-Critical Systems Framework

This repository provides the theoretical and computational base for the near-critical research line.

## Core model

The system is a capacity-constrained stochastic buffer with random replenishment disruptions and Poisson demand:

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

Small positive `delta` defines the near-critical regime: the system is stable in expectation but vulnerable to collapse under disruption sequences.

## Paper A contribution

Paper A frames disruption-driven collapse as a first-passage reliability problem. It studies corrected inverse-Gaussian first-passage dynamics, hazard structure, and reliability-based threshold control.

Main theoretical components:

- first-passage collapse modeling;
- corrected inverse-Gaussian shape parameter;
- hazard-structure characterization;
- decreasing-failure-rate interpretation via survivorship selection;
- average-cost Markov decision process;
- reliability-based threshold intervention.

## Connection to later repositories

```text
Near-Critical Systems / Paper A
        ↓
Controlled Near-Critical Benchmark / Paper B
        ↓
Supply Chain Digital Twin application
```

The present repository is the foundational theory and reproducibility package. The Paper B repository documents the controlled near-critical benchmark package, and the Digital Twin repository documents the industrial/networked application.
