from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
PLOTS_DIR = Path('plots')
OUTPUT_BASENAME = 'ED4_mental_model_full_figure'
SCENARIOS = ['CLOSE', 'ASV', 'BIKE']
TEXT_X_LABELS = ['Before', 'After', 'GT']
TEXT_Y_LABELS = ['Before', 'After']
TEXT_RATIONALE_FILES = {'Experts': [DATA_DIR / 'expert_text_rationale_llm_judgments.csv', RESULTS_DIR / 'expert_text_rationale_llm_judgments.csv'], 'Non-experts': [DATA_DIR / 'non_expert_text_rationale_llm_judgments.csv', RESULTS_DIR / 'non_expert_text_rationale_llm_judgments.csv']}
CORRELATION_FILES = {'Experts': DATA_DIR / 'expert_simulator_responses.csv', 'Non-experts': DATA_DIR / 'non_expert_simulator_responses.csv'}
Q_MAP = {'a': 'ASV', 'b': 'BIKE', 'c': 'CLOSE'}
GROUND_TRUTH = {'a': 2, 'b': 2, 'c': 2}
CORR_X_ORDER = ['Worse', 'Same', 'Better']
CORR_Y_ORDER = ['Worse', 'Better']

def find_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    candidates = '\n  '.join((str(path) for path in paths))
    raise FileNotFoundError(f'Could not find {label}. Looked for:\n  {candidates}')

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

def mixed_effects_annotation(subset: pd.DataFrame) -> str:
    if subset.empty or subset['T1_Numeric'].nunique() < 2 or subset['T2_Numeric'].nunique() < 2:
        return 'Mixed effects:\nN/A'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = smf.mixedlm('T2_Numeric ~ T1_Numeric', subset, groups=subset['Participant_ID'])
            result = model.fit(reml=False, method='lbfgs', maxiter=500, disp=False)
        coefficient = result.params.get('T1_Numeric', np.nan)
        p_value = result.pvalues.get('T1_Numeric', np.nan)
        return f'Mixed effects:\nCoeff: {coefficient:.2f}, P-val: {p_value_text(p_value)}'
    except Exception:
        try:
            result = smf.ols('T2_Numeric ~ T1_Numeric', subset).fit(cov_type='cluster', cov_kwds={'groups': subset['Participant_ID']})
            coefficient = result.params.get('T1_Numeric', np.nan)
            p_value = result.pvalues.get('T1_Numeric', np.nan)
            return f'Clustered OLS:\nCoeff: {coefficient:.2f}, P-val: {p_value_text(p_value)}'
        except Exception:
            return 'Mixed effects:\nModel error'

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
    ax.text(0.03, 0.97, mixed_effects_annotation(subset), transform=ax.transAxes, ha='left', va='top', fontsize=6, bbox={'boxstyle': 'round,pad=0.2', 'facecolor': '#ffffee', 'edgecolor': '0.65', 'alpha': 0.95})

def make_figure() -> plt.Figure:
    expert_results = find_existing(TEXT_RATIONALE_FILES['Experts'], 'expert text-rationale results')
    non_expert_results = find_existing(TEXT_RATIONALE_FILES['Non-experts'], 'non-expert text-rationale results')
    expert_matrices = load_text_rationale_matrices(expert_results)
    non_expert_matrices = load_text_rationale_matrices(non_expert_results)
    df_experts_corr = process_correlation_file(CORRELATION_FILES['Experts'], 'Experts', participant_offset=0)
    df_non_experts_corr = process_correlation_file(CORRELATION_FILES['Non-experts'], 'Non-experts', participant_offset=1000)
    df_corr_all = pd.concat([df_experts_corr, df_non_experts_corr], ignore_index=True)
    if df_corr_all.empty:
        raise ValueError('No valid rows were created from the correlation input data.')
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
    fig = make_figure()
    pdf_path = PLOTS_DIR / f'{OUTPUT_BASENAME}.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {pdf_path}')
if __name__ == '__main__':
    main()
