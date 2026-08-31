"""Reproduce the Extended Data mental-model figure and its statistics.

(Cited as Extended Data Fig. 4 in the paper body text; the corresponding
caption in the current proof is Extended Data Fig. 5 - see the README.)

Figure (plots/ED4_mental_model_full_figure.pdf):
  Rows A/B: confusion matrices of free-form text-rationale alignment for
    experts (A, N=9) and non-experts (B, N=30). Each participant's before/after
    responses were matched by an LLM judge (GPT-5, validated against human
    raters; see reproduce_SI_LLM_Judge.py) against three anchors: the on-road
    safety driver's belief before the explanation, after the explanation, and
    the ground truth.
  Row C: heatmaps correlating nearest-neighbour (NN) score changes with text
    rationale changes, annotated with OLS regressions using participant-
    clustered standard errors.

Statistics log (logs/mental_model_alignment_stats.log):
  - the confusion-matrix counts,
  - the exact binomial tests for the paper's Hypotheses 1-3 (two-sided,
    H0: p=0.5 between the before/after rows, per anchor),
  - the per-group clustered OLS coefficients shown in row C,
  - the group x NN-score interaction OLS reported in the figure caption.

Inputs: data/{expert,non_expert}_text_rationale_llm_judgments.csv and
data/{expert,non_expert}_simulator_responses.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import binomtest
DATA_DIR = Path('data')
PLOTS_DIR = Path('plots')
LOGS_DIR = Path('logs')
OUTPUT_BASENAME = 'ED4_mental_model_full_figure'
STATS_LOG_PATH = LOGS_DIR / 'mental_model_alignment_stats.log'
SCENARIOS = ['CLOSE', 'ASV', 'BIKE']
TEXT_X_LABELS = ['Before', 'After', 'GT']
TEXT_Y_LABELS = ['Before', 'After']
ANCHOR_LABELS = ['Before explanation', 'After explanation', 'Ground truth']
TEXT_RATIONALE_FILES = {'Experts': DATA_DIR / 'expert_text_rationale_llm_judgments.csv', 'Non-experts': DATA_DIR / 'non_expert_text_rationale_llm_judgments.csv'}
CORRELATION_FILES = {'Experts': DATA_DIR / 'expert_simulator_responses.csv', 'Non-experts': DATA_DIR / 'non_expert_simulator_responses.csv'}
Q_MAP = {'a': 'ASV', 'b': 'BIKE', 'c': 'CLOSE'}  # survey question block -> scenario
GROUND_TRUTH = {'a': 2, 'b': 2, 'c': 2}  # correct nearest-neighbour answer per block
CORR_X_ORDER = ['Worse', 'Same', 'Better']
CORR_Y_ORDER = ['Worse', 'Better']

def find_existing(path: Path, label: str) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f'Could not find {label} at {path}')

def scenario_matrix(df: pd.DataFrame, scenarios: list[str]) -> np.ndarray:
    matrix = np.zeros((2, 3), dtype=int)
    for scenario in scenarios:
        if scenario not in df.columns:
            raise KeyError(f'Missing required column {scenario!r} in text-rationale results file.')
    for _, row in df.iterrows():
        for scenario in scenarios:
            choices = str(row[scenario]).strip().split()
            if len(choices) != 3:
                raise ValueError(f'Expected three choices in column {scenario!r}, got {row[scenario]!r}')
            for anchor_idx, choice in enumerate(choices):
                try:
                    row_idx = int(choice) - 1
                except ValueError as exc:
                    raise ValueError(f'Expected choice 1 or 2, got {choice!r}') from exc
                if row_idx not in (0, 1):
                    raise ValueError(f'Expected choice 1 or 2, got {choice!r}')
                matrix[row_idx, anchor_idx] += 1
    return matrix

def load_text_rationale_matrices(csv_path: Path) -> list[tuple[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    matrices = [('All scenarios', scenario_matrix(df, SCENARIOS))]
    matrices.extend(((f'{scenario} scenario', scenario_matrix(df, [scenario])) for scenario in SCENARIOS))
    return matrices

def spectrum_score(answer: object, confidence: object, correct_answer: int) -> float | None:
    try:
        if pd.isna(answer) or pd.isna(confidence):
            return None
        answer_int = int(float(answer))
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        return None
    return confidence_float if answer_int == int(correct_answer) else -confidence_float

def score_change_category(delta: float) -> tuple[str, int]:
    if delta > 0:
        return ('Better', 1)
    if delta < 0:
        return ('Worse', -1)
    return ('Same', 0)

def rationale_change_category(value: object) -> tuple[str, int] | None:
    try:
        value_int = int(float(value))
    except (TypeError, ValueError):
        return None
    if value_int == 1:
        return ('Worse', 0)
    if value_int == 2:
        return ('Better', 1)
    return None

def process_correlation_file(filepath: Path, group_name: str, participant_offset: int) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f'Could not find required correlation file: {filepath}')
    df = pd.read_csv(filepath)
    rows: list[dict[str, object]] = []
    for q_char, mm_prefix in Q_MAP.items():
        gt_answer = GROUND_TRUTH[q_char]
        required_columns = [f'{q_char}1', f'{q_char}2_1', f'{q_char}6', f'{q_char}7_1', f'{mm_prefix} gt MM']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise KeyError(f'{filepath} is missing required columns: {missing}')
        for participant_idx, row in df.iterrows():
            start_score = spectrum_score(row[f'{q_char}1'], row[f'{q_char}2_1'], gt_answer)
            end_score = spectrum_score(row[f'{q_char}6'], row[f'{q_char}7_1'], gt_answer)
            if start_score is None or end_score is None:
                continue
            task_1_cat, task_1_num = score_change_category(end_score - start_score)
            rationale_cat = rationale_change_category(row[f'{mm_prefix} gt MM'])
            if rationale_cat is None:
                continue
            task_2_cat, task_2_num = rationale_cat
            rows.append({'Participant_ID': participant_idx + participant_offset, 'Group': group_name, 'Task_1_Cat': task_1_cat, 'Task_2_Cat': task_2_cat, 'T1_Numeric': task_1_num, 'T2_Numeric': task_2_num})
    return pd.DataFrame(rows)

def p_value_text(p_value: float) -> str:
    if np.isnan(p_value):
        return 'nan'
    if p_value < 0.001:
        return '0.000'
    return f'{p_value:.3f}'

def fit_clustered_ols(subset: pd.DataFrame):
    """OLS of rationale change on NN score change with participant-clustered
    standard errors (the specification reported in the figure caption)."""
    return smf.ols('T2_Numeric ~ T1_Numeric', subset).fit(cov_type='cluster', cov_kwds={'groups': subset['Participant_ID']})

def clustered_ols_annotation(subset: pd.DataFrame) -> str:
    if subset.empty or subset['T1_Numeric'].nunique() < 2 or subset['T2_Numeric'].nunique() < 2:
        return 'Clustered OLS:\nN/A'
    result = fit_clustered_ols(subset)
    coefficient = result.params.get('T1_Numeric', np.nan)
    p_value = result.pvalues.get('T1_Numeric', np.nan)
    return f'Clustered OLS:\nCoeff: {coefficient:.2f}, P-val: {p_value_text(p_value)}'

def draw_heatmap(ax: plt.Axes, matrix: np.ndarray, x_labels: list[str], y_labels: list[str], title: str, xlabel: str, ylabel: str | None=None, vmin: float | None=None, vmax: float | None=None, colorbar_label: str='Count', annotation_size: int=10) -> None:
    image = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8, rotation=90, va='center')
    ax.set_xlabel(xlabel, fontsize=8.5, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, labelpad=2)
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='both', length=2, pad=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center', fontsize=annotation_size, fontweight='bold', color='black')
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

def draw_text_rationale_row(axes: list[plt.Axes], matrices: list[tuple[str, np.ndarray]], first_ylabel: bool) -> None:
    for idx, (ax, (title, matrix)) in enumerate(zip(axes, matrices)):
        draw_heatmap(ax=ax, matrix=matrix, x_labels=TEXT_X_LABELS, y_labels=TEXT_Y_LABELS, title=title, xlabel='On-road safety driver belief', ylabel='Online study\nparticipant belief' if idx == 0 and first_ylabel else None, vmin=None, vmax=None, colorbar_label='Count', annotation_size=10)

def draw_correlation_heatmap(ax: plt.Axes, df_all: pd.DataFrame, group: str) -> None:
    subset = df_all[df_all['Group'] == group].copy()
    counts = pd.crosstab(subset['Task_2_Cat'], subset['Task_1_Cat'])
    counts = counts.reindex(index=CORR_Y_ORDER, columns=CORR_X_ORDER, fill_value=0)
    matrix = counts.to_numpy(dtype=int)
    draw_heatmap(ax=ax, matrix=matrix, x_labels=CORR_X_ORDER, y_labels=CORR_Y_ORDER, title=group, xlabel='Task 1: NN score change', ylabel='Task 2: Rationale change', vmin=0, vmax=max(1, int(matrix.max())), colorbar_label='Count', annotation_size=10)
    ax.set_aspect('equal', adjustable='box')
    ax.text(0.03, 0.97, clustered_ols_annotation(subset), transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': '#ffffee', 'edgecolor': '0.65', 'alpha': 0.95})

def format_matrix_lines(matrix: np.ndarray) -> list[str]:
    lines = [f"{'':>8}  {'Before':>8}  {'After':>8}  {'GT':>8}"]
    for row_idx, row_label in enumerate(TEXT_Y_LABELS):
        lines.append(f'{row_label:>8}  ' + '  '.join((f'{int(matrix[row_idx, col]):>8}' for col in range(matrix.shape[1]))))
    return lines

def binomial_test_lines(matrix: np.ndarray) -> list[str]:
    """Two-sided exact binomial tests, one per anchor (column), of the split
    between the Before row and the After row under H0: p = 0.5.

    On the non-expert all-scenarios matrix these are the paper's Hypotheses
    1-3 (P < 1e-9, P < 0.0002, and P < 1e-5, respectively).
    """
    lines = []
    for anchor_idx, anchor_label in enumerate(ANCHOR_LABELS):
        successes = int(matrix[0, anchor_idx])
        failures = int(matrix[1, anchor_idx])
        n = successes + failures
        p_value = binomtest(successes, n, p=0.5, alternative='two-sided').pvalue if n else float('nan')
        lines.append(f'  anchor={anchor_label!r}: counts=[{successes}, {failures}], n={n}, two-sided exact binomial p={p_value:.4g}')
    return lines

def write_statistics_log(matrices_by_group: dict[str, list[tuple[str, np.ndarray]]], df_corr_all: pd.DataFrame, log_path: Path=STATS_LOG_PATH) -> Path:
    lines: list[str] = []
    lines.append('Mental model alignment statistics')
    lines.append('=' * 33)
    lines.append('')
    lines.append('Confusion matrices and exact binomial tests')
    lines.append('-------------------------------------------')
    lines.append('Rows: participant belief matched to the before/after response;')
    lines.append('columns: safety-driver anchor (before / after explanation / ground truth).')
    lines.append('Each test is a two-sided exact binomial test of the before/after split')
    lines.append('within one column under H0: p = 0.5.')
    lines.append('The three tests on the non-expert "All scenarios" matrix are the exact')
    lines.append("binomial tests for the paper's Hypotheses 1-3.")
    for group_name, matrices in matrices_by_group.items():
        for title, matrix in matrices:
            lines.append('')
            lines.append(f'{group_name} - {title}')
            lines.extend(format_matrix_lines(matrix))
            lines.extend(binomial_test_lines(matrix))
    lines.append('')
    lines.append('Clustered OLS: text rationale change ~ NN score change (row C annotations)')
    lines.append('---------------------------------------------------------------------------')
    for group_name in ['Experts', 'Non-experts']:
        subset = df_corr_all[df_corr_all['Group'] == group_name]
        result = fit_clustered_ols(subset)
        beta = result.params['T1_Numeric']
        se = result.bse['T1_Numeric']
        p_value = result.pvalues['T1_Numeric']
        lines.append(f"{group_name}: beta = {beta:.4f}, SE = {se:.4f}, p = {p_value:.4f} (n = {len(subset)} observations, {subset['Participant_ID'].nunique()} participants)")
    lines.append('')
    lines.append('Group x NN score change interaction (figure caption)')
    lines.append('-----------------------------------------------------')
    lines.append('OLS with participant-clustered standard errors; group coded as')
    lines.append('non-experts = 1, experts = 0.')
    interaction_df = df_corr_all.copy()
    interaction_df['grp'] = (interaction_df['Group'] == 'Non-experts').astype(int)
    interaction_result = smf.ols('T2_Numeric ~ T1_Numeric * grp', interaction_df).fit(cov_type='cluster', cov_kwds={'groups': interaction_df['Participant_ID']})
    beta = interaction_result.params['T1_Numeric:grp']
    se = interaction_result.bse['T1_Numeric:grp']
    p_value = interaction_result.pvalues['T1_Numeric:grp']
    lines.append(f'Interaction T1_Numeric:group: beta = {beta:.4f}, SE = {se:.4f}, p = {p_value:.4f}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return log_path

def load_inputs() -> tuple[dict[str, list[tuple[str, np.ndarray]]], pd.DataFrame]:
    expert_results = find_existing(TEXT_RATIONALE_FILES['Experts'], 'expert text-rationale results')
    non_expert_results = find_existing(TEXT_RATIONALE_FILES['Non-experts'], 'non-expert text-rationale results')
    matrices_by_group = {'Experts': load_text_rationale_matrices(expert_results), 'Non-experts': load_text_rationale_matrices(non_expert_results)}
    # The participant offset keeps expert and non-expert cluster IDs distinct
    # when both groups are pooled for the interaction model.
    df_experts_corr = process_correlation_file(CORRELATION_FILES['Experts'], 'Experts', participant_offset=0)
    df_non_experts_corr = process_correlation_file(CORRELATION_FILES['Non-experts'], 'Non-experts', participant_offset=1000)
    df_corr_all = pd.concat([df_experts_corr, df_non_experts_corr], ignore_index=True)
    if df_corr_all.empty:
        raise ValueError('No valid rows were created from the correlation input data.')
    return (matrices_by_group, df_corr_all)

def make_figure(matrices_by_group: dict[str, list[tuple[str, np.ndarray]]], df_corr_all: pd.DataFrame) -> plt.Figure:
    expert_matrices = matrices_by_group['Experts']
    non_expert_matrices = matrices_by_group['Non-experts']
    fig = plt.figure(figsize=(13.2, 8.0))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.22], left=0.1, right=0.985, bottom=0.08, top=0.94, wspace=0.3, hspace=0.48)
    axes_a = [fig.add_subplot(grid[0, col]) for col in range(4)]
    axes_b = [fig.add_subplot(grid[1, col]) for col in range(4)]
    axes_c = [fig.add_subplot(grid[2, 1]), fig.add_subplot(grid[2, 2])]
    draw_text_rationale_row(axes_a, expert_matrices, first_ylabel=True)
    draw_text_rationale_row(axes_b, non_expert_matrices, first_ylabel=True)
    draw_correlation_heatmap(axes_c[0], df_corr_all, 'Experts')
    draw_correlation_heatmap(axes_c[1], df_corr_all, 'Non-experts')
    fig.text(0.055, 0.958, 'A', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.055, 0.63, 'B', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.055, 0.311, 'C', fontsize=15, fontweight='bold', ha='left', va='top')
    fig.text(0.025, 0.785, 'Experts', fontsize=13, fontweight='bold', rotation=90, ha='center', va='center')
    fig.text(0.025, 0.465, 'Non-experts', fontsize=13, fontweight='bold', rotation=90, ha='center', va='center')
    fig.text(0.025, 0.175, 'Mental model\nmetrics', fontsize=13, fontweight='bold', rotation=90, ha='center', va='center')
    return fig

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    matrices_by_group, df_corr_all = load_inputs()
    fig = make_figure(matrices_by_group, df_corr_all)
    pdf_path = PLOTS_DIR / f'{OUTPUT_BASENAME}.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')
    log_path = write_statistics_log(matrices_by_group, df_corr_all)
    print(f'Saved statistics log to {log_path}')
if __name__ == '__main__':
    main()
