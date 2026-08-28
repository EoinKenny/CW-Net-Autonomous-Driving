from __future__ import annotations
import re
from pathlib import Path
from statistics import NormalDist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
DATA_DIR = Path('data')
PLOTS_DIR = Path('plots')
LOGS_DIR = Path('logs')
CONTROL_CSV = 'sagat_control_responses.csv'
EXPERIMENTAL_CSV = 'sagat_experimental_responses.csv'
GROUND_TRUTH_CSV = 'sagat_ground_truth_answers.csv'
APPLY_ATTENTION_CHECK = True
APPLY_TIMING_FILTERS = False
FIGURE_PATH = PLOTS_DIR / 'overall_valence_accuracy.pdf'
LOG_PATH = LOGS_DIR / 'overall_valence_results.log'
QUESTION_ORDER = ['perception', 'comprehension', 'projection']
GROUP_ORDER = ['Control', 'Experimental']
VALENCE_CODE = {'positive': 'p', 'negative': 'n'}
VALENCE_LABEL = {'positive': 'Surprising events', 'negative': 'Unsurprising events'}
VALENCE_TITLES = {'positive': 'Surprising events\n(mirroring the private track tests)', 'negative': 'Unsurprising events\n(Checking robustness of explanations)'}
THRESHOLDS = {'cp11': 8, 'cn11': 4, 'ap11': 16, 'an11': 16, 'bp11': 25, 'bn11': 15, 'ac11': 15, 'cp21': 45, 'cn21': 5, 'ap21': 18, 'an21': 15, 'bp21': 15, 'bn21': 10}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols: dict[str, str] = {}
    current_group: str | None = None
    for col in df.columns:
        page_submit = re.match('^t?([a-z]{2})(\\d+)_Page Submit$', str(col))
        if page_submit:
            letters, number = page_submit.groups()
            current_group = f'{letters}{number}'
            new_cols[col] = f'{current_group}_Page Submit'
            continue
        numbered_question = re.match('^Q\\d+_([1-6])$', str(col))
        if numbered_question and current_group:
            new_cols[col] = f'{current_group}_{numbered_question.group(1)}'
            continue
        letter_question = re.match('^([a-z]{2})\\d+_([1-6])$', str(col))
        if letter_question and current_group:
            letters, question = letter_question.groups()
            if current_group.startswith(letters):
                new_cols[col] = f'{current_group}_{question}'
                continue
        already_normalized = re.match('^[a-z]{2}\\d+_[1-6]$', str(col))
        if already_normalized:
            new_cols[col] = str(col)
            continue
        new_cols[col] = str(col)
    df = df.rename(columns=new_cols)
    keep = [col for col in df.columns if col == 'Q37' or re.match('^[a-z]{2}\\d+_Page Submit$', str(col)) or re.match('^[a-z]{2}\\d+_[1-6]$', str(col))]

    def sort_key(col: str) -> tuple[str, int, int]:
        if col == 'Q37':
            return ('aa', -1, -1)
        match = re.match('^([a-z]{2})(\\d+)_(Page Submit|[1-6])$', str(col))
        if not match:
            return ('zz', 10 ** 9, 10 ** 9)
        letters, number, tail = match.groups()
        tail_order = 0 if tail == 'Page Submit' else int(tail)
        return (letters, int(number), tail_order)
    return df[sorted(keep, key=sort_key)]

def process_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    if ground_truth.empty:
        raise ValueError(f'{GROUND_TRUTH_CSV} is empty.')
    normalized_answer_cols = [col for col in ground_truth.columns if re.match('^[a-z]{2}\\d+_[1-6]$', str(col)) and (not str(col).startswith('ac1_'))]
    has_qualtrics_timing_cols = any(('Page Submit' in str(col) or 'First Click' in str(col) or 'Last Click' in str(col) or ('Click Count' in str(col)) for col in ground_truth.columns))
    if normalized_answer_cols and (not has_qualtrics_timing_cols):
        one_row = ground_truth.iloc[[0]].copy().reset_index(drop=True)
        return one_row[normalized_answer_cols]
    if len(ground_truth) < 3:
        raise ValueError(f'Could not parse {GROUND_TRUTH_CSV}. It is neither an already-processed one-row file with columns like an1_1/ap2_6, nor a raw Qualtrics file with at least three rows.')
    one_row = ground_truth.iloc[[2]].copy().reset_index(drop=True)
    normalized = normalize_columns(one_row)
    answer_cols = [col for col in normalized.columns if re.match('^[a-z]{2}\\d+_[1-6]$', str(col)) and (not str(col).startswith('ac1_'))]
    if not answer_cols:
        raise ValueError('Could not find any ground-truth answer columns after normalization. Expected columns like an1_1, ap2_6, cp1_5, etc.')
    return normalized[answer_cols]

def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').astype(float)
    text = series.astype(str).str.strip()
    as_timedelta = pd.to_timedelta(text, errors='coerce').dt.total_seconds()
    as_numeric = pd.to_numeric(text, errors='coerce')
    return as_timedelta.where(as_timedelta.notna(), as_numeric).astype(float)

def filter_attention_check(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['ac1_3', 'ac1_4', 'ac1_5', 'ac1_6']
    if not set(cols).issubset(df.columns):
        return df.reset_index(drop=True)
    values = df[cols].apply(pd.to_numeric, errors='coerce')
    keep = (values['ac1_3'] == 1) & (values['ac1_4'] == 2) & (values['ac1_5'] == 1) & (values['ac1_6'] == 2)
    return df.loc[keep].reset_index(drop=True)

def filter_timing_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for key, min_seconds in THRESHOLDS.items():
        col = f'{key}_Page Submit'
        if col in df.columns:
            keep &= _to_seconds(df[col]).ge(min_seconds).fillna(False)
    return df.loc[keep].reset_index(drop=True)

def should_drop_qualtrics_metadata_rows(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    candidate_cols = [col for col in df.columns if re.match('^[a-z]{2}\\d+_[1-6]$', str(col)) or re.match('^[a-z]{2}\\d+_Page Submit$', str(col))]
    if not candidate_cols:
        return False
    first_two = df.loc[df.index[:2], candidate_cols]
    later = df.loc[df.index[2:], candidate_cols]
    first_numeric_rate = pd.to_numeric(first_two.stack(), errors='coerce').notna().mean()
    later_numeric_rate = pd.to_numeric(later.stack(), errors='coerce').notna().mean()
    return bool(first_numeric_rate < 0.25 and later_numeric_rate > 0.5)

def process_response_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.dropna(how='all').copy()
    df = normalize_columns(df)
    if should_drop_qualtrics_metadata_rows(df):
        df = df.iloc[2:].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    answer_cols = [col for col in df.columns if re.match('^[a-z]{2}\\d+_[1-6]$', str(col))]
    if answer_cols:
        df = df.dropna(subset=answer_cols, how='all').reset_index(drop=True)
    return df

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Expected input file not found: {path}\nPut {CONTROL_CSV}, {EXPERIMENTAL_CSV}, and {GROUND_TRUTH_CSV} in {DATA_DIR}/.')
    df = pd.read_csv(path)
    # Some Qualtrics exports are prefixed by a single dataset-name line before
    # the actual comma-separated header. In that case pandas sees one column
    # and silently places each subsequent CSV row into the index.
    if len(df.columns) == 1:
        with path.open('r', encoding='utf-8-sig') as handle:
            first_line = handle.readline()
            second_line = handle.readline()
        if ',' not in first_line and ',' in second_line:
            df = pd.read_csv(path, skiprows=1)
    return df

def load_and_clean_data(data_dir: Path=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_con_raw = read_csv_required(data_dir / CONTROL_CSV)
    df_exp_raw = read_csv_required(data_dir / EXPERIMENTAL_CSV)
    ground_truth_raw = read_csv_required(data_dir / GROUND_TRUTH_CSV)
    df_con = process_response_dataframe(df_con_raw)
    df_exp = process_response_dataframe(df_exp_raw)
    ground_truth_df = process_ground_truth(ground_truth_raw)
    if APPLY_ATTENTION_CHECK:
        df_con = filter_attention_check(df_con)
        df_exp = filter_attention_check(df_exp)
    if APPLY_TIMING_FILTERS:
        df_con = filter_timing_thresholds(df_con)
        df_exp = filter_timing_thresholds(df_exp)
    return (df_con, df_exp, ground_truth_df)

def equal_answer(a: object, b: object) -> bool:
    a_num = pd.to_numeric(pd.Series([a]), errors='coerce').iloc[0]
    b_num = pd.to_numeric(pd.Series([b]), errors='coerce').iloc[0]
    if pd.notna(a_num) and pd.notna(b_num):
        return bool(a_num == b_num)
    if pd.isna(a) or pd.isna(b):
        return False
    return str(a).strip().casefold() == str(b).strip().casefold()

def infer_groups(ground_truth_df: pd.DataFrame) -> list[str]:
    return sorted({match.group(1) for col in ground_truth_df.columns for match in [re.match('^([a-z]{2}\\d+)_[1-6]$', str(col))] if match})

def select_groups_by_valence(ground_truth_df: pd.DataFrame, valence: str) -> list[str]:
    code = VALENCE_CODE[valence]
    groups = [group for group in infer_groups(ground_truth_df) if len(group) >= 2 and group[1] == code]
    if not groups:
        raise ValueError(f'No groups found for valence={valence!r}.')
    return groups

def compute_accuracies(df: pd.DataFrame, ground_truth_df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    ground_truth = ground_truth_df.iloc[0].to_dict()
    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        perception_hits: list[int] = []
        comprehension_hits: list[int] = []
        projection_hits: list[int] = []
        for group in groups:
            for question in range(1, 5):
                col = f'{group}_{question}'
                if col in ground_truth and col in row:
                    perception_hits.append(int(equal_answer(row[col], ground_truth[col])))
            col = f'{group}_5'
            if col in ground_truth and col in row:
                comprehension_hits.append(int(equal_answer(row[col], ground_truth[col])))
            col = f'{group}_6'
            if col in ground_truth and col in row:
                projection_hits.append(int(equal_answer(row[col], ground_truth[col])))
        rows.append({'perception': float(np.mean(perception_hits)) if perception_hits else np.nan, 'comprehension': float(np.mean(comprehension_hits)) if comprehension_hits else np.nan, 'projection': float(np.mean(projection_hits)) if projection_hits else np.nan})
    return pd.DataFrame(rows)

def stderr(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

def summarize_for_plot(df_con: pd.DataFrame, df_exp: pd.DataFrame, ground_truth_df: pd.DataFrame, valence: str) -> pd.DataFrame:
    groups = select_groups_by_valence(ground_truth_df, valence)
    control = compute_accuracies(df_con, ground_truth_df, groups)
    control['group'] = 'Control'
    experimental = compute_accuracies(df_exp, ground_truth_df, groups)
    experimental['group'] = 'Experimental'
    long = pd.concat([control, experimental], ignore_index=True).melt(id_vars='group', value_vars=QUESTION_ORDER, var_name='question_type', value_name='accuracy').dropna(subset=['accuracy'])
    summary = long.groupby(['group', 'question_type'], as_index=False).agg(mean=('accuracy', 'mean'), sem=('accuracy', stderr))
    summary['question_type'] = pd.Categorical(summary['question_type'], categories=QUESTION_ORDER, ordered=True)
    summary['group'] = pd.Categorical(summary['group'], categories=GROUP_ORDER, ordered=True)
    return summary.sort_values(['question_type', 'group'])

def plot_overall_by_valence(df_con: pd.DataFrame, df_exp: pd.DataFrame, ground_truth_df: pd.DataFrame, output_path: Path=FIGURE_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    x = np.arange(len(QUESTION_ORDER))
    width = 0.35
    for ax, valence in zip(axes, ['positive', 'negative']):
        summary = summarize_for_plot(df_con, df_exp, ground_truth_df, valence)
        for i, group in enumerate(GROUP_ORDER):
            group_summary = summary[summary['group'] == group]
            ax.bar(x + (i - 0.5) * width, group_summary['mean'], width, yerr=group_summary['sem'], capsize=4, label=group)
        ax.set_xticks(x)
        ax.set_xticklabels(QUESTION_ORDER, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_title(VALENCE_TITLES[valence], fontsize=16, pad=14)
    axes[0].legend(loc='upper right', frameon=True)
    axes[1].legend(loc='lower left', frameon=True)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return output_path

def cohens_d_with_ci(control: pd.Series, experimental: pd.Series, alpha: float=0.05) -> tuple[float, float, float, int, int]:
    control_values = pd.Series(control).dropna().to_numpy(dtype=float)
    experimental_values = pd.Series(experimental).dropna().to_numpy(dtype=float)
    n_control = len(control_values)
    n_experimental = len(experimental_values)
    if n_control < 2 or n_experimental < 2:
        return (np.nan, np.nan, np.nan, n_control, n_experimental)
    control_sd = control_values.std(ddof=1)
    experimental_sd = experimental_values.std(ddof=1)
    pooled_sd = np.sqrt(((n_control - 1) * control_sd ** 2 + (n_experimental - 1) * experimental_sd ** 2) / (n_control + n_experimental - 2))
    if pooled_sd == 0:
        return (0.0, 0.0, 0.0, n_control, n_experimental)
    d = (experimental_values.mean() - control_values.mean()) / pooled_sd
    se_d = np.sqrt((n_control + n_experimental) / (n_control * n_experimental) + d ** 2 / (2 * (n_control + n_experimental)))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return (d, d - z * se_d, d + z * se_d, n_control, n_experimental)

def effect_size_label(d: float) -> str:
    if pd.isna(d):
        return 'n/a'
    abs_d = abs(d)
    if abs_d < 0.2:
        return 'negligible'
    if abs_d < 0.5:
        return 'small'
    if abs_d < 0.8:
        return 'medium'
    return 'large'

def p_value_label(p_value: float) -> str:
    if pd.isna(p_value):
        return 'n/a'
    if p_value < 0.001:
        return '< 0.001'
    return f'{p_value:.3f}'

def effect_direction(d: float, p_value: float, alpha: float=0.05) -> str:
    if pd.isna(d) or pd.isna(p_value) or p_value >= alpha:
        return 'No effect'
    return 'Exp > Control' if d > 0 else 'Control > Exp'

def calculate_statistics(df_con: pd.DataFrame, df_exp: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    bonferroni_alpha = 0.05 / 6
    for valence in ['positive', 'negative']:
        groups = select_groups_by_valence(ground_truth_df, valence)
        control = compute_accuracies(df_con, ground_truth_df, groups).dropna()
        experimental = compute_accuracies(df_exp, ground_truth_df, groups).dropna()
        for question_type in QUESTION_ORDER:
            control_scores = control[question_type].dropna()
            experimental_scores = experimental[question_type].dropna()
            d, ci_low, ci_high, n_control, n_experimental = cohens_d_with_ci(control_scores, experimental_scores)
            t_stat, p_value = stats.ttest_ind(experimental_scores, control_scores, equal_var=False, nan_policy='omit')
            rows.append({'valence': valence, 'valence_label': VALENCE_LABEL[valence], 'question_type': question_type, 'control_mean': float(control_scores.mean()), 'control_sd': float(control_scores.std(ddof=1)), 'experimental_mean': float(experimental_scores.mean()), 'experimental_sd': float(experimental_scores.std(ddof=1)), 'difference_exp_minus_control': float(experimental_scores.mean() - control_scores.mean()), 't_statistic': float(t_stat), 'p_value': float(p_value), 'bonferroni_alpha': bonferroni_alpha, 'bonferroni_significant': bool(p_value < bonferroni_alpha), 'cohens_d_exp_minus_control': float(d), 'effect_size_label': effect_size_label(d), 'ci_low': float(ci_low), 'ci_high': float(ci_high), 'effect_direction': effect_direction(d, p_value), 'n_control': int(n_control), 'n_experimental': int(n_experimental)})
    return pd.DataFrame(rows)

def write_statistics_log(results: pd.DataFrame, output_path: Path=LOG_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['SAGAT overall valence statistics', '================================', "Sign convention: positive Cohen's d = Experimental higher than Control.", "Cohen's d CI: 95% normal-approximation CI using the large-sample variance of d.", 'p-value: Welch independent-samples t-test comparing Experimental vs Control.', 'Bonferroni correction: alpha = 0.05 / 6 = 0.008333.', 'Effect direction uses the uncorrected p < 0.05 direction; Bonferroni significance is reported separately.', '']
    for valence in ['positive', 'negative']:
        valence_results = results[results['valence'] == valence]
        lines.append(VALENCE_LABEL[valence])
        lines.append('-' * len(VALENCE_LABEL[valence]))
        for _, row in valence_results.iterrows():
            bonf = 'Yes' if row['bonferroni_significant'] else 'No'
            lines.extend([f"Dimension: {row['question_type'].capitalize()}", f"  Effect direction: {row['effect_direction']}", f"  Control:      M = {row['control_mean']:.3f}, SD = {row['control_sd']:.3f}, n = {int(row['n_control'])}", f"  Experimental: M = {row['experimental_mean']:.3f}, SD = {row['experimental_sd']:.3f}, n = {int(row['n_experimental'])}", f"  Difference:   {row['difference_exp_minus_control']:+.3f}", f"  t-statistic:  {row['t_statistic']:+.3f}", f"  p-value:      {p_value_label(row['p_value'])}", f"  Cohen's d:    {row['cohens_d_exp_minus_control']:+.3f} ({row['effect_size_label']})", f"  95% CI:       [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]", f"  Bonferroni significant at alpha={row['bonferroni_alpha']:.6f}: {bonf}", ''])
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path

def main() -> None:
    df_con, df_exp, ground_truth_df = load_and_clean_data(DATA_DIR)
    figure_path = plot_overall_by_valence(df_con, df_exp, ground_truth_df, FIGURE_PATH)
    results = calculate_statistics(df_con, df_exp, ground_truth_df)
    log_path = write_statistics_log(results, LOG_PATH)
    print(f'Saved figure to {figure_path}')
    print(f'Saved statistics log to {log_path}')
if __name__ == '__main__':
    main()
