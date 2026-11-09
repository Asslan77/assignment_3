# Sports facility pricing and demand forecasting

**Kuandyk Bibarys and Tulebay Aslan**, Astana IT University.

## Files

- `paper/`: full article PDF, LaTeX source and bibliography.
- `data/`: dataset, generator settings and hidden parameters used only for evaluation.
- `src/`: generation, analysis, plotting and verification scripts.
- `results/`: predictions, metrics, figures and execution details.
- `results_pipeline.ipynb`: notebook for inspecting and reproducing the run.

## Run

Use Python 3.12 in a virtual environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py
python src/check_artifacts.py
python src/build_results_section.py
```

The analysis uses a CPU and overwrites derived results. The recorded run took about six minutes. Exact versions and hardware are in `environment.json`; another platform may need a compatible PyTorch installation.

## License

- Code: MIT License (see `LICENSE`).
- Data (`data/`): CC BY 4.0 (see `LICENSE-DATA`).

## Citation and archive

Archived release: https://doi.org/10.5281/zenodo.24080033 (v1.0)
