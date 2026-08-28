# CW-Net Reproducibility Capsule — Autonomous Driving

This repository contains all code, data, and materials to reproduce the results from the CW-Net paper on concept-based explanations for autonomous driving.

![Overview of the CW-Net architecture and autonomous-driving evaluation pipeline.](images/cwnet.png)

*Figure: Overview of the CW-Net architecture and how it was visualized to safety drivers and engineers.*



## Repository Structure

```
.
├── scripts/                      # Scripts for reproducing the paper figures
├── data/                         # Anonymised study data for reproducing the figures
├── code_ocean_capsule/           # Demo in a toy domain (training + visualization);
│                                 #   submitted as the Nature Code Ocean capsule
├── qualtrics_survey/             # Qualtrics surveys for running the online studies
├── original_cwnet_model_files/   # Original CW-Net model definitions that ran on the car
├── reproduce_paper_results.py    # Main runner (orchestrates all scripts)
├── plots/                        # Generated figures (created on first run)
└── logs/                         # Execution logs (created on first run)
```

## Setup

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate cwnet
```

If that doesn't work try using pip directly:

```bash
conda create -n cwnet python=3.10 -y
conda activate cwnet
pip install -r requirements.txt
```

### Reproduce all paper results

```bash
python reproduce_paper_results.py
```

Figures are saved to `plots/` and logs to `logs/`.


#### Options

```bash
# Run specific scripts only
python reproduce_paper_results.py --only reproduce_figure3.py reproduce_figureED3.py

# Skip specific scripts
python reproduce_paper_results.py --skip reproduce_SI_LLM_Judge.py

# Continue past failures
python reproduce_paper_results.py --continue-on-error

# Preview what would run
python reproduce_paper_results.py --dry-run
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical arrays and linear algebra |
| pandas | Data loading and manipulation |
| matplotlib | Figure generation |
| seaborn | Statistical visualizations |
| scipy | Statistical tests, distance metrics, optimization |
| statsmodels | Linear mixed-effects models |
| fastdtw | Dynamic Time Warping (optional; graceful fallback) |



## License

This repository contains code, anonymised data, and user-study materials. These are licensed separately.

- Source code is licensed under the Apache License, Version 2.0. See `LICENSE.txt`.
- Anonymised data files in `data/` are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See `LICENSE-DATA.txt`.
- User-study materials, including video stimuli showing vehicle driving scenes with overlaid concept activations, are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License unless otherwise stated. See `LICENSE-MATERIALS.txt`.



## Citation and acknowledgements

This repository accompanies the paper:

> Kenny et al., "Explainable deep learning improves human mental models of self-driving cars", Nature, 2026.

This work was conducted in collaboration with Motional, the Massachusetts Institute of Technology, and Harvard University.

Please cite the paper when using the code, data, or study materials from this repository.



## Data and materials release

The released data and materials are intended to support reproducibility of the published analyses. The release includes anonymised CSV files and user-study video stimuli used in the experiments.

The release does not include raw vehicle logs, raw sensor streams, internal model checkpoints, participant identifiers, reviewer names, or confidential Motional engineering materials.