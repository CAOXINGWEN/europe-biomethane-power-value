# Non-fossil sinks and spatial lock-in erode the climate value of European food-waste biomethane power

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24361422.svg)](https://doi.org/10.5281/zenodo.24361422)

**Authors:** Xingwen Cao$^{1,2}$, Zhengguang Liu$^{3}$, Hailong Li$^{4}$, Haoran Zhang$^{1,2,\ast}$ & Lingyue Wang$^{5}$

1. School of Urban Planning and Design, Peking University Shenzhen Graduate School, Shenzhen, Guangdong 518055, China.
2. Guangdong Provincial Key Laboratory of Energy Transition Risk Perception and Sustainable Governance, Shenzhen, Guangdong 518055, China.
3. Department of Chemical Engineering, University of Manchester, Manchester M13 9PL, UK.
4. Future Energy Center, Mälardalen University, 72123 Västerås, Sweden.
5. School of Information Management, Xinjiang University of Finance and Economics, Urumqi, 830012, China.

$\ast$ **Correspondence:** Haoran Zhang (`h.zhang@pku.edu.cn`)

---

## Overview

This repository provides the complete processed source datasets, an open-source paired linear-programming (LP) dispatch benchmark, publication figure-generation scripts, an independent numerical verification suite, and the multi-stage model code supporting all findings in **"Non-fossil sinks and spatial lock-in erode the climate value of European food-waste biomethane power"**.

Key analytical components covered in this repository:
- **Pan-European multi-horizon paired dispatch (2028–2035):** 15 countries (24 market zones) across 36 historical weather years (8,760 hourly steps per year) under the ENTSO-E 2025 European Resource Adequacy Assessment (*National Trends*) trajectory.
- **Causal factorial decompositions:** Exact Shapley and Möbius attributions for fixed-2035-feedstock $2^3$ system-mechanism experiments (wind/solar availability $V$, fossil/carbon dispatch cost $C$, and system background $B$) and $2^2$ scale-versus-composition experiments (7.30 TWh Central food waste vs 151.41 TWh existing biogas fleet).
- **Operational biowaste plant benchmarks:** Full-year 8,760-hour paired dispatch, nodal capacity swaps, and measured cogeneration-versus-upgrading parity thresholds for **Pinerolo** (northern Italy, zone `ITN1`) and **Berlin Ruhleben** (Germany, zone `DE00`).
- **Dual energy-closure & parity boundaries:** Deliverable-biomethane and equal-raw-biogas closure bases across 110,592 equal-leakage cases, 270 joint asymmetric-leakage cells in 2035, and 36 constrained grid-access rollout scenarios.

---

## Repository Structure

| Path | Description |
|---|---|
| `source_data/` | All CSV, JSON, NPY, and GeoJSON datasets underlying Figs. 1–4, Extended Data Figs. 1–10, Supplementary Fig. S1, and Supplementary Tables S1–S6, plus `Source_Data_v17.xlsx` and `data_dictionary.csv`. |
| `example/` | Self-contained 168-hour two-stage lexicographic LP replay benchmark (`replay.py` + compressed `.npz` inputs) using only open-source `SciPy` (`HiGHS` dual-simplex solver) with zero commercial solver license required. |
| `scripts/` | Standalone scripts to reproduce all 15 figures (`reproduce_all_figures.py`, `main_figures.py`, `extended_data_figures.py`) and verify all headline numbers, Shapley/Möbius attributions, and plant benchmarks (`verify_results.py`). |
| `figures/` | Pre-rendered vector PDFs and high-resolution PNGs for Figs. 1–4, Extended Data Figs. 1–10, and Supplementary Fig. S1. |
| `model_code/` | Full Python model and post-processing scripts across all experimental stages (`00_adequacy_siting_v35` through `04_facility_benchmarks_de_it`), indexed in `model_code/code_manifest.csv`. |
| `CITATION.cff` / `.zenodo.json` | Machine-readable citation and Zenodo archival metadata. |
| `SHA256SUMS.txt` | Cryptographic SHA-256 checksums of every file in this repository. |

---

## Quickstart (Under 1 Minute)

### 1. Set up Python environment

Using `pip`:
```bash
pip install -r requirements.txt
```
Or using `conda` / `mamba`:
```bash
conda env create -f environment.yml
conda activate biomethane-grid-v17
```

### 2. Run the open-source 168-hour paired LP dispatch benchmark

Solves stage-1 unserved-energy minimization and stage-2 dispatch-cost minimization via `scipy.optimize.linprog(method="highs-ds")` for both Pinerolo (`ITN1`) and pan-European injection (April 2035, weather year WS17) in ~5 seconds:
```bash
python example/replay.py
```

### 3. Verify all manuscript numbers and exact Shapley/Möbius attributions

Recomputes headline 36-weather trajectories, exact $2^3$ and $2^2$ Shapley/Möbius decompositions, Berlin Ruhleben & Pinerolo annual benchmarks, and the 270-cell / 36-rollout parity boundaries directly from `source_data/`:
```bash
python scripts/verify_results.py
```

### 4. Reproduce all Main, Extended Data, and Supplementary figures

Regenerates Figs. 1–4, Extended Data Figs. 1–10, and Supplementary Fig. S1 (exporting PDF, SVG, PNG, and 600-dpi TIFF):
```bash
python scripts/reproduce_all_figures.py
```

---

## Full-Horizon Dispatch Campaigns (`model_code/`)

Full 8,760-hour multi-year campaign batches (`1,273` runs in `02_decarbonization_trajectory_v37`, `121` factorial runs in `03_factorial_and_boundaries_v38`, and full-year weekly/monthly anchored site runs in `04_facility_benchmarks_de_it`) were executed on Linux workstations using Cardinal Optimizer (`coptpy`, COPT v8.0) for multi-threaded barrier/dual-simplex throughput and cross-checked with open-source `SciPy HiGHS`. Raw public ENTSO-E ERAA 2025 climate-year time series (~840 MB across 24 zones and 36 weather years) are obtainable directly from [ENTSO-E ERAA 2025](https://www.entsoe.eu/outlooks/eraa/).

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 52472316, 52341203, and 52461160297); the Science and Technology Strategic Consulting Project of the Chinese Academy of Engineering (Project No. 2025-HZ-31); the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2025A1515010251); the Shenzhen Natural Science Foundation in Basic Research Fund (Grant No. JCYJ20250604175822030); the Research Project of the State-owned Assets Supervision and Administration Commission of the State Council (SASAC) (Project No. SASAC-RC2026-10); the Key-Area Research and Development Program of Guangdong Province (Grant No. 2025B0909040001); the Shenzhen Science and Technology Program (Project No. KQTD20221101093604016); and the Natural Science Foundation of Xinjiang Uygur Autonomous Region (Grant No. 2026D01C342). This work was also supported by the High-performance Computing Platform of Peking University.

---

## License & Citation

Code is released under the **MIT License** and processed source datasets/figures under **CC BY 4.0** (see `LICENSE`). Please cite the accompanying paper and `CITATION.cff` when using these materials.
