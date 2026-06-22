# Archived results

## Beta calibration

Source: `results/beta_calibration/summary_n15000_T300000.json`

| Quantity | Definitive value |
|---|---:|
| Design points | 42 |
| Replications per point | 15,000 |
| Horizon | 300,000 |
| `a` | 0.2477150491 |
| `b` | 0.3929272836 |
| `c` | 0.0111266596 |
| Raw-beta R-squared | 0.9535754625 |
| Mean uncorrected eta error | 36.0126% |
| Mean corrected eta error | 5.0110% |
| Maximum uncorrected eta error | 62.6378% |
| Maximum corrected eta error | 18.3830% |
| Mean-error improvement | 7.1868 times |

Eta-level relative error is normalized by the fitted shape parameter:
`100*|eta_prediction-eta_fitted|/eta_fitted`.

## Policy and reliability tables

- `diagnostic_hazard_paper.csv`: 21 feasible configurations; approximate DFR
  in all 21 and strict smoothed DFR in 11.
- `exp_cost_ratio_paper.csv`: five failure/prevention cost ratios; maximum
  improvement over the age-based benchmark is approximately 21.56%.
- `exp_heatmap_paper.csv`: operating-grid threshold and cost results; 21 valid
  configurations.
- `exp_misspecification_paper.csv`: parameter-misspecification audit; maximum
  valid regret is approximately 5.32%.
- `exp_smax_paper.csv` and `exp_smax_summary_paper.csv`: capacity sensitivity.
- `lemma1_audit_paper.csv`: the conditional lemma holds in all 21 valid
  configurations.

The archived quick and legacy beta fits are deliberately excluded because they
are not the calibration reported in the submitted manuscript.
