# SADO: supplementary code and data

Reproduces every number, table and figure in the manuscript.

* `sim.py` – slot-level edge-IoT offloading simulator and all policies (baselines, IQL, SADO and ablations).
* `experiments.py` – runs the full held-out evaluation (seeds 101–110; E1 main, E2 staleness, E3 load, E4 report phase, E5 scale, E6 ablation). Resumable; writes `results/runs.jsonl`.
* `analyze.py` – computes means, 95% CIs and paired Wilcoxon tests, writes the LaTeX tables, figures and `numbers.tex` (the macros used for every number in the text).
* `verify.py` – independent recomputation of 58 key numbers from the raw runs, checked against `numbers.tex`.
* `results/runs.jsonl` – one JSON record per run (655 runs); `results/summary.json` – aggregated statistics.

Requirements: Python 3.10+, NumPy, SciPy, pandas, matplotlib. Paths at the top of the scripts point to the original working directory; edit `OUT`, `R` and `F` as needed. Development seeds 1–3 were used only for design selection and are not part of `runs.jsonl`.
