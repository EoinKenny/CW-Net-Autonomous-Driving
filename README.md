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

### Run the Code Ocean toy demo

The Code Ocean capsule uses a separate Python 3.9 environment and includes
all required training data. The commands below work on macOS and Linux. Run
them from the repository root.

#### 1. Create the local Python environment

This only needs to be done once:

```bash
conda create --prefix "$PWD/.venv-codeocean" --override-channels --channel conda-forge python=3.9 pip=22.1.2 swig -y
conda activate "$PWD/.venv-codeocean"
python -m pip install --disable-pip-version-check -r code_ocean_capsule/code/requirements.txt
```

The environment is created at `.venv-codeocean/` and is ignored by git. pip
22.1.2 is intentional because newer pip versions reject Gym 0.24.0's legacy
package metadata.

#### 2. Run the capsule

Activate the environment if it is not already active, then run `main.py`:

```bash
conda activate "$PWD/.venv-codeocean"
cd code_ocean_capsule/code
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 python -u main.py
```

The run trains CW-Net for 50 epochs and evaluates it in three CarRacing
simulations. It takes a few minutes and prints `Successful training!` when
the mean simulation reward exceeds 200. A successful run should have a final
training MSE around 0.25.

Outputs are created in `code_ocean_capsule/results/`, including:

- `simulation.gif` — the first simulation with predicted concepts
- `accuracy.pdf`, `losses.pdf`, and `confusion_matrix_concept_accuracy.pdf`
- `rewards.npy` and `error.npy`

The setup helper `bash code_ocean_capsule/code/setup.sh` can be used instead
of the three environment-creation commands above.

### Reproduce all paper results

```bash
python reproduce_paper_results.py
```

Figures are saved to `plots/` and statistics logs to `logs/` (both created on
first run). The full run takes well under a minute on a laptop and is fully
deterministic. The individual scripts read data via relative paths, so run
them from the repository root (the runner does this automatically).

#### Script-to-results mapping

| Script | Paper result | Outputs | Notes |
|--------|--------------|---------|-------|
| `reproduce_figure3.py` | Fig. 3a/b/c (on-road CLOSE, ASV, BIKE results) | `plots/CLOSE_*.pdf`, `plots/ASV_*.pdf`, `plots/BIKE_*.pdf` | Fig. 3c retains the cyclist runs and excludes the pedestrian-only windows. Its `n = 23` is the pooled total of 14 retained before-explanation and 9 after-explanation cyclist trials; the caption wording can read as though all 23 belong to the second round. For Fig. 3a, the caption reports CLOSE `n = 5,545`, whereas the released data/code give `n = 5,543`, a two-row filter-boundary difference. |
| `reproduce_figureED3.py` | Extended Data Fig. 3 belief-update ("arrows") figure | `plots/combined_movement_2x2.pdf` | Each arrow represents one participant: 9 experts and 30 non-experts, averaged across 3 scenarios. The caption's `N = 27` and `N = 90` count participant-scenario observations. The generated PDF states this explicitly. |
| `reproduce_figureED4.py` | Extended Data Fig. 4 mental-model figure (text-rationale confusion matrices + NN/rationale correlation); exact binomial tests for Hypotheses 1–3; clustered OLS + interaction from the caption | `plots/ED4_mental_model_full_figure.pdf`, `logs/mental_model_alignment_stats.log` | The paper reports the expert clustered-OLS value as `P = 0.243`; the released analysis gives `P = 0.241`. Both are non-significant; this minor numerical/rounding difference does not affect the interpretation. |
| `reproduce_figureED5_analysis.py` | Extended Data Fig. 5 mental-model-vs-prediction LME figure and caption statistics | `plots/Comparison_Task1_vs_Task2_Stacked_Reordered.pdf`, `logs/Comparison_Task1_vs_Task2_Stacked_Reordered_LME.log` | The expert nearest-neighbour effect (beta = 2.02, SE = 0.87, P = 0.021) uses ordinal coding because the categorical model is singular: no expert observation falls in the `Worsened` reference level. The non-expert effect (beta = 9.86, SE = 2.07, P < 0.001) uses categorical `Improved`-versus-`Worsened` coding. The generated PDF and log state this explicitly. |
| `reproduce_figureED7_analysis.py` | Extended Data Fig. 7 SAGAT results figure; Welch t-tests and Cohen's d (Methods, "Public roads evaluation using SAGAT") | `plots/overall_valence_accuracy.pdf`, `logs/overall_valence_results.log` | - |
| `reproduce_SI_LLM_Judge.py` | LLM-as-a-judge vs human raters interrater validation (SI) | `logs/llm_human_interrater_validation.txt` | - |
| `reproduce_SI_cronbach_alpha.py` | Supplementary Tables S8 and S9 (SAGAT reliability) | `logs/reproduce_SI_cronbach_alpha.log` | - |

Not reproducible from this repository (per the paper's Data Availability
statement): Extended Data Table 1 and Supplementary Tables 1 and 2 (they
require internal AV model outputs that cannot be released). The complete
training data for the open-source Code Ocean toy demo is included under
`code_ocean_capsule/data/`; see `code_ocean_capsule/README.txt` for its
separate setup and run instructions.


#### Options

```bash
# Run specific scripts only
python reproduce_paper_results.py --only reproduce_figure3.py reproduce_figureED3.py

# Skip specific scripts
python reproduce_paper_results.py --skip reproduce_SI_LLM_Judge.py reproduce_SI_cronbach_alpha.py

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
| fastdtw | Dynamic Time Warping (required for the Fig. 3b DTW panels) |



## License

This repository contains code, anonymised data, and user-study materials. These are licensed separately.

- Source code is licensed under the Apache License, Version 2.0. See `LICENSE.txt`.
- Anonymised data files in `data/` are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See `LICENSE-DATA.txt`.
- The toy-domain training data in `code_ocean_capsule/data/` is released under CC0 1.0 Universal. See `code_ocean_capsule/data/LICENSE`.
- User-study materials, including video stimuli showing vehicle driving scenes with overlaid concept activations, are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License unless otherwise stated. See `LICENSE-MATERIALS.txt`.



## Citation and acknowledgements

This repository accompanies the paper:

> Kenny et al., "Explainable deep learning improves human mental models of self-driving cars", Nature, 2026.

This work was conducted in collaboration with Motional, the Massachusetts Institute of Technology, and Harvard University.

Please cite the paper when using the code, data, or study materials from this repository.



## Data and materials release

The released data and materials are intended to support reproducibility of the published analyses. The release includes anonymised CSV files and user-study video stimuli used in the experiments.

The release does not include raw vehicle logs, raw sensor streams, internal model checkpoints, participant identifiers, reviewer names, or confidential Motional engineering materials.
