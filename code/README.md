# RealLit — Quick Start

CPU-only end-to-end demo of the **RealLit** benchmark. Loads a subset cohort, runs
one cell through the same pipeline the paper uses (impute → standardize
→ classify → ROC-AUC), and aggregates folds into a per-cell summary.

> **Reproducing the paper's full evaluation grid?**
> See [`deploy/runpod/README.md`](deploy/runpod/README.md) for installing
> the paper methods (TabPFN-v2, TabICL-v2, TabDPT, MaskMLP, MIRI,
> TabCSDI, DiffPuter, CFMI), HuggingFace weights, the Dockerfile, and
> RunPod Serverless deployment.

## Install

```bash
git clone https://github.com/anonymous-submission-site/real-lit.git
cd real-lit/code
pip install -e ".[dev]"
```

Python 3.10+. Core deps: `numpy`, `pandas`, `scikit-learn`. **No torch,
no GPU, no model weights** for the quick start.

## Run the demo

```bash
python scripts/examples/run_minimal.py
```

Runs **5 folds** of D2 / LWR (KTEA-3 Letter-Word Recognition at-risk
target) with MEAN imputation and HGB (sklearn HistGradientBoosting)
under stratified k-fold CV; aggregates folds into one per-cell row.
**~10 s on a laptop, AUC ≈ 0.91.**

Output:
```
results/minimal_demo/D2/LWR/none/0/MEAN/HGB/seed{0..4}.json
results/minimal_demo/ablation_results.csv
```

The per-fold JSON follows the project's full schema (see
`scripts/run_cell.py` docstring): `cell`, `data_shape`, `metrics`
(classification + imputation), `subgroup_metrics` per demographic axis,
`timing`, `provenance`, `config`.

## Run one cell manually

```bash
python scripts/run_cell.py \
    --dataset D2 --target LWR \
    --regime MCAR --rate 30 \
    --imputer MEAN --classifier HGB \
    --K 1 --seed 0 \
    --run-name myrun
```

`HGB` and `LogReg` are sklearn-only stand-in classifiers (not in the
paper) so the harness produces real numbers without GPU. The four
**paper classifiers** are scaffold stubs that need their reference
implementations wired in — see [`deploy/runpod/README.md`](deploy/runpod/README.md).

## Layout

```
src/n26/         package: data, classifiers, imputers, missingness, metrics
scripts/         run_cell.py, aggregate.py, regenerate_matrix.py, examples/
docs/            ablation_matrix.csv (the paper's full grid)
deploy/runpod/   Docker + serverless handler (full-scale reproduction)
tests/           pytest suite
```

## License

CC BY 4.0. See [LICENSE](LICENSE).
