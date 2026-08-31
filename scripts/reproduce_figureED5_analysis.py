"""Reproduce the Extended Data mental-model-vs-prediction LME figure/analysis.

(Cited as Extended Data Fig. 5 in the paper body text; the corresponding
caption in the current proof is Extended Data Fig. 6 - see the README.)

Figure (plots/Comparison_Task1_vs_Task2_Stacked_Reordered.pdf): prediction
confidence deltas by mental-model category (nearest-neighbour task and text
rationale), for experts and non-experts.

Log (logs/Comparison_Task1_vs_Task2_Stacked_Reordered_LME.log): linear
mixed-effects models (participant random intercepts) relating prediction
confidence change to mental-model change.

Model-specification note: for experts, the categorical NN model
(Improved/No change/Worsened, reference = Worsened) is singular because no
expert observation falls in the 'Worsened' reference level; the log records
this as MODEL FAILED. The expert NN effect reported in the paper caption
(beta = 2.02, SE = 0.87, p = 0.02) therefore comes from the ordinal coding
(Worsened=-1, No change=0, Improved=+1), whereas the non-expert NN effect
(beta = 9.86, SE = 2.07) comes from the categorical model. The
CAPTION-STYLE PRIMARY EFFECTS section at the end of the log lists exactly
the models behind the caption values.

Inputs: data/expert_simulator_responses.csv and
data/non_expert_simulator_responses.csv.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
DATA_DIR = Path('data')
PLOTS_DIR = Path('plots')
LOGS_DIR = Path('logs')
PLOT_FILENAME = 'Comparison_Task1_vs_Task2_Stacked_Reordered.pdf'
LOG_FILENAME = 'Comparison_Task1_vs_Task2_Stacked_Reordered_LME.log'
Q_MAP = {'a': 'ASV', 'b': 'BIKE', 'c': 'CLOSE'}
GT_TASK1 = {'a': 2, 'b': 2, 'c': 2}
GT_TASK2 = {'a': 2, 'b': 2, 'c': 1}
FILES = {'Experts': 'expert_simulator_responses.csv', 'Non-Experts': 'non_expert_simulator_responses.csv'}
ROW_ORDER = ['Non-Experts', 'Experts']
DOT_PALETTE = {'Prediction improved': '#006400', 'Prediction worsened': '#8B0000', 'Prediction unchanged': '#404040'}
TEXT_PALETTE = {'Improved': '#e6ffe6', 'Worsened': '#ffe6e6'}
B1_PALETTE = {'Improved': '#e6ffe6', 'Worsened': '#ffe6e6', 'No change': '#f0f0f0'}
ORDER_TEXT = ['Worsened', 'Improved']
ORDER_B1 = ['Worsened', 'No change', 'Improved']
BIN_BLOCK1_NUMERIC = {'Worsened': -1, 'No change': 0, 'Improved': 1}
BIN_TEXT_NUMERIC = {'Worsened': 0, 'Improved': 1}

def get_spectrum_score(answer: object, confidence: object, correct_answer: int) -> Optional[float]:
    if pd.isna(answer) or pd.isna(confidence):
        return None
    try:
        answer_int = int(float(answer))
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        return None
    return confidence_float if answer_int == int(correct_answer) else -confidence_float

def get_prediction_change_status(start_answer: object, end_answer: object, correct_answer: int) -> str:
    try:
        start_correct = int(float(start_answer)) == int(correct_answer)
        end_correct = int(float(end_answer)) == int(correct_answer)
    except (TypeError, ValueError):
        return 'Error'
    if not start_correct and end_correct:
        return 'Prediction improved'
    if start_correct and (not end_correct):
        return 'Prediction worsened'
    return 'Prediction unchanged'

def prepare_group_data(df: pd.DataFrame) -> pd.DataFrame:
    plot_data = []
    for q_char, mm_prefix in Q_MAP.items():
        gt_1 = GT_TASK1[q_char]
        gt_2 = GT_TASK2[q_char]
        required_columns = [f'{q_char}1', f'{q_char}2_1', f'{q_char}3', f'{q_char}4_1', f'{q_char}6', f'{q_char}7_1', f'{q_char}8', f'{q_char}9_1', f'{mm_prefix} gt MM']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError('Missing required columns: ' + ', '.join(missing_columns))
        for participant_id, row in df.iterrows():
            b1_start = get_spectrum_score(row[f'{q_char}1'], row[f'{q_char}2_1'], gt_1)
            b1_end = get_spectrum_score(row[f'{q_char}6'], row[f'{q_char}7_1'], gt_1)
            b2_start = get_spectrum_score(row[f'{q_char}3'], row[f'{q_char}4_1'], gt_2)
            b2_end = get_spectrum_score(row[f'{q_char}8'], row[f'{q_char}9_1'], gt_2)
            if any((value is None for value in [b1_start, b1_end, b2_start, b2_end])):
                continue
            score_delta_block1 = b1_end - b1_start
            if score_delta_block1 > 0:
                bin_block1 = 'Improved'
            elif score_delta_block1 < 0:
                bin_block1 = 'Worsened'
            else:
                bin_block1 = 'No change'
            try:
                text_mm_value = int(float(row[f'{mm_prefix} gt MM']))
            except (TypeError, ValueError):
                continue
            if text_mm_value == 2:
                bin_text = 'Improved'
            elif text_mm_value == 1:
                bin_text = 'Worsened'
            else:
                continue
            plot_data.append({'Participant_ID': participant_id, 'Question_ID': q_char, 'Score_Delta_Block1': score_delta_block1, 'Score_Delta_Block2': b2_end - b2_start, 'Prediction_Change_Block2': get_prediction_change_status(row[f'{q_char}3'], row[f'{q_char}8'], gt_2), 'Bin_Block1': bin_block1, 'Bin_Block1_Numeric': BIN_BLOCK1_NUMERIC[bin_block1], 'Bin_Text': bin_text, 'Bin_Text_Numeric': BIN_TEXT_NUMERIC[bin_text]})
    return pd.DataFrame(plot_data)

def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    viz_dfs = {}
    for group_name, filename in FILES.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f'Could not find expected file: {csv_path}')
        raw_df = pd.read_csv(csv_path)
        viz_dfs[group_name] = prepare_group_data(raw_df)
    return viz_dfs

def plot_comparison(viz_dfs: dict[str, pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16 / 1.3, 14 / 1.3), sharey=True)
    for row_idx, group_name in enumerate(ROW_ORDER):
        if group_name not in viz_dfs or viz_dfs[group_name].empty:
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)
            continue
        v_df = viz_dfs[group_name]
        sns.boxplot(data=v_df, x='Bin_Block1', y='Score_Delta_Block2', order=ORDER_B1, ax=axes[row_idx, 0], palette=B1_PALETTE, showfliers=False)
        sns.swarmplot(data=v_df, x='Bin_Block1', y='Score_Delta_Block2', hue='Prediction_Change_Block2', order=ORDER_B1, palette=DOT_PALETTE, ax=axes[row_idx, 0], size=6, alpha=0.8)
        axes[row_idx, 0].set_ylabel(f'{group_name}\nPrediction confidence delta', fontsize=12, fontweight='bold')
        axes[row_idx, 0].set_xlabel('Mental model category')
        axes[row_idx, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[row_idx, 0].set_title(f'{group_name}: NN score vs prediction', fontsize=14)
        legend = axes[row_idx, 0].get_legend()
        if legend is not None:
            legend.remove()
        sns.boxplot(data=v_df, x='Bin_Text', y='Score_Delta_Block2', order=ORDER_TEXT, ax=axes[row_idx, 1], palette=TEXT_PALETTE, showfliers=False)
        sns.swarmplot(data=v_df, x='Bin_Text', y='Score_Delta_Block2', hue='Prediction_Change_Block2', order=ORDER_TEXT, palette=DOT_PALETTE, ax=axes[row_idx, 1], size=6, alpha=0.8)
        axes[row_idx, 1].set_ylabel('')
        axes[row_idx, 1].set_xlabel('Mental model category')
        axes[row_idx, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[row_idx, 1].set_title(f'{group_name}: Text rationale vs prediction', fontsize=14)
        legend = axes[row_idx, 1].get_legend()
        if row_idx == 0:
            axes[row_idx, 1].legend(loc='upper left', bbox_to_anchor=(1, 1), title='Prediction change')
        elif legend is not None:
            legend.remove()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return '= NA'
    if p_value < 0.001:
        return '< 0.001'
    return f'= {p_value:.3f}'

def fit_mixedlm(formula: str, df: pd.DataFrame):
    model = smf.mixedlm(formula, df, groups=df['Participant_ID'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        try:
            return model.fit()
        except Exception:
            pass
        last_error = None
        for method in ('lbfgs', 'bfgs', 'cg', 'powell', 'nm'):
            try:
                return model.fit(method=method, maxiter=2000, disp=False)
            except Exception as error:
                last_error = error
    raise RuntimeError(f"MixedLM fit failed for formula '{formula}': {last_error}")

def find_param_name(result, contains: str) -> Optional[str]:
    for param_name in result.params.index:
        if contains in param_name:
            return param_name
    return None

def summarize_effect(result, param_name: str) -> str:
    beta = result.params[param_name]
    se = result.bse[param_name]
    p_value = result.pvalues[param_name]
    return f'beta = {beta:.2f}, SE = {se:.2f}, p {format_p_value(p_value)}'

def add_model_block(lines: list[str], title: str, formula: str, df: pd.DataFrame, primary_param_contains: str) -> Optional[object]:
    lines.append('')
    lines.append(title)
    lines.append('-' * len(title))
    lines.append(f'Formula: {formula}')
    lines.append(f'Observations: {len(df)}')
    lines.append(f"Participants: {df['Participant_ID'].nunique()}")
    lines.append('Category counts:')
    for column in ['Bin_Block1', 'Bin_Text', 'Prediction_Change_Block2']:
        if column in df.columns:
            counts = df[column].value_counts(dropna=False).to_dict()
            lines.append(f'  {column}: {counts}')
    try:
        result = fit_mixedlm(formula, df)
    except Exception as error:
        lines.append(f'MODEL FAILED: {error}')
        return None
    primary_param = find_param_name(result, primary_param_contains)
    if primary_param is None:
        lines.append(f'Primary effect not found. Looked for: {primary_param_contains}')
        lines.append(f'Available parameters: {list(result.params.index)}')
    else:
        lines.append(f'Primary effect ({primary_param}): {summarize_effect(result, primary_param)}')
    lines.append('')
    lines.append(str(result.summary()))
    return result

def run_lme_analysis(viz_dfs: dict[str, pd.DataFrame], log_path: Path) -> str:
    lines: list[str] = []
    lines.append('Task 1 vs Task 2 LME analysis')
    lines.append('=' * 31)
    lines.append('')
    lines.append('Outcome: Score_Delta_Block2')
    lines.append('  = signed prediction confidence after explanation')
    lines.append('    minus signed prediction confidence before explanation.')
    lines.append('')
    lines.append('Signed confidence is positive when the prediction is correct')
    lines.append('and negative when the prediction is incorrect.')
    lines.append('')
    lines.append('Random effects: participant-level random intercepts.')
    lines.append('')
    lines.append('Primary caption-style categorical models:')
    lines.append('  NN score model: Worsened is the reference; primary effect is Improved vs Worsened.')
    lines.append('  Text rationale model: Worsened is the reference; primary effect is Improved vs Worsened.')
    lines.append('  Where a categorical model is singular (no observations in the reference')
    lines.append('  level), the ordinal-coding model (Worsened=-1, No change=0, Improved=+1)')
    lines.append('  provides the caption value instead; this applies to the Experts NN model.')
    caption_effects: list[str] = []
    for group_name in ['Experts', 'Non-Experts']:
        if group_name not in viz_dfs or viz_dfs[group_name].empty:
            lines.append('')
            lines.append(f'Skipping {group_name}: no usable data.')
            continue
        df = viz_dfs[group_name].copy()
        df['Bin_Block1'] = pd.Categorical(df['Bin_Block1'], categories=ORDER_B1, ordered=True)
        df['Bin_Text'] = pd.Categorical(df['Bin_Text'], categories=ORDER_TEXT, ordered=True)
        lines.append('')
        lines.append('=' * 80)
        lines.append(group_name.upper())
        lines.append('=' * 80)
        group_effects: list[str] = []
        nn_formula = "Score_Delta_Block2 ~ C(Bin_Block1, Treatment(reference='Worsened'))"
        nn_result = add_model_block(lines=lines, title=f'{group_name}: NN category vs prediction delta', formula=nn_formula, df=df, primary_param_contains='[T.Improved]')
        if nn_result is not None:
            primary_param = find_param_name(nn_result, '[T.Improved]')
            if primary_param is not None:
                group_effects.append(f'{group_name} NN category, Improved vs Worsened: {summarize_effect(nn_result, primary_param)}')
        text_formula = "Score_Delta_Block2 ~ C(Bin_Text, Treatment(reference='Worsened'))"
        text_result = add_model_block(lines=lines, title=f'{group_name}: text rationale category vs prediction delta', formula=text_formula, df=df, primary_param_contains='[T.Improved]')
        if text_result is not None:
            primary_param = find_param_name(text_result, '[T.Improved]')
            if primary_param is not None:
                group_effects.append(f'{group_name} text rationale, Improved vs Worsened: {summarize_effect(text_result, primary_param)}')
        ordinal_formula = 'Score_Delta_Block2 ~ Bin_Block1_Numeric'
        ordinal_result = add_model_block(lines=lines, title=f'{group_name}: ordinal NN category check', formula=ordinal_formula, df=df, primary_param_contains='Bin_Block1_Numeric')
        if nn_result is None and ordinal_result is not None:
            # The categorical NN model is singular when the 'Worsened' reference
            # level has no observations (this happens for experts); the paper
            # caption reports the ordinal-coding effect for that group instead.
            primary_param = find_param_name(ordinal_result, 'Bin_Block1_Numeric')
            if primary_param is not None:
                group_effects.insert(0, f"{group_name} NN category (ordinal coding; categorical model is singular because no observation falls in the 'Worsened' reference level): {summarize_effect(ordinal_result, primary_param)}")
        caption_effects.extend(group_effects)
        raw_delta_formula = 'Score_Delta_Block2 ~ Score_Delta_Block1'
        add_model_block(lines=lines, title=f'{group_name}: raw NN signed-confidence delta check', formula=raw_delta_formula, df=df, primary_param_contains='Score_Delta_Block1')
    lines.append('')
    lines.append('=' * 80)
    lines.append('CAPTION-STYLE PRIMARY EFFECTS')
    lines.append('=' * 80)
    if caption_effects:
        lines.extend(caption_effects)
    else:
        lines.append('No primary effects could be estimated.')
    log_text = '\n'.join(lines) + '\n'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding='utf-8')
    return log_text

def main() -> None:
    viz_dfs = load_data(DATA_DIR)
    plot_path = PLOTS_DIR / PLOT_FILENAME
    log_path = LOGS_DIR / LOG_FILENAME
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plot_comparison(viz_dfs, plot_path)
        run_lme_analysis(viz_dfs, log_path)
    print(f'Saved figure to {plot_path.resolve()}')
    print(f'Saved LME log to {log_path.resolve()}')
if __name__ == '__main__':
    main()
