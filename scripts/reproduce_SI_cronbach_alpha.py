"""Reproduce Supplementary Tables S8 and S9 (SAGAT reliability analysis).

Table S8: Cronbach's alpha for the SAGAT perception scale (Q1-Q4) and its
AV-state (Q2, Q4) and world-state (Q1, Q3) subscales, computed both
classically and ordinally. Because the items are binary, the ordinal alpha
uses tetrachoric inter-item correlations (estimated by inverting the
bivariate-normal CDF with Brent's method, with a 0.5 continuity correction
for empty cells).

Table S9: per-question pooled accuracy, mean item variance, and the number of
near-ceiling items (mean accuracy >= 0.85), which diagnoses why the classical
alpha is deflated.

Participant filtering matches the paper's Methods (attention checks and
minimum viewing times; see THRESHOLDS). This preprocessing was written
independently of reproduce_figureED7_analysis.py but was verified to select
the identical n = 48 control + n = 51 experimental participants; the two
groups are pooled here.

Inputs: data/sagat_{control,experimental}_responses.csv and
data/sagat_ground_truth_answers.csv.
Output: logs/reproduce_SI_cronbach_alpha.log (markdown tables).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm
DATA_DIR = Path('data')
LOG_DIR = Path('logs')
LOG_FILE = LOG_DIR / 'reproduce_SI_cronbach_alpha.log'
CEILING_THRESHOLD = 0.85
# Minimum plausible per-page viewing times in seconds (see the docstring in
# reproduce_figureED7_analysis.py for the key naming scheme).
THRESHOLDS = {'cp11': 8, 'cn11': 4, 'ap11': 16, 'an11': 16, 'bp11': 25, 'bn11': 15, 'ac11': 15, 'cp21': 45, 'cn21': 5, 'ap21': 18, 'an21': 15, 'bp21': 15, 'bn21': 10}

def first_existing(data_dir: Path, names: Iterable[str]) -> Path:
    candidates = [data_dir / name for name in names]
    for path in candidates:
        if path.exists():
            return path
    expected = '\n  - '.join((str(p) for p in candidates))
    raise FileNotFoundError(f'Could not find any of these input files:\n  - {expected}')

def read_csv_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some Qualtrics exports include a dataset-name wrapper line before the
    # actual comma-separated header. Retry from the real header when detected.
    if len(df.columns) == 1:
        with path.open('r', encoding='utf-8-sig') as handle:
            first_line = handle.readline()
            second_line = handle.readline()
        if ',' not in first_line and ',' in second_line:
            df = pd.read_csv(path, skiprows=1)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    current_group = None
    for col in df.columns:
        m_ps = re.match('^t([a-z]{2})(\\d+)_Page Submit$', col)
        if m_ps:
            letters, num = m_ps.groups()
            current_group = f'{letters}{num}'
            new_cols[col] = f'{current_group}_Page Submit'
            continue
        m_ps2 = re.match('^([a-z]{2})(\\d+)_Page Submit$', col)
        if m_ps2:
            letters, num = m_ps2.groups()
            current_group = f'{letters}{num}'
            new_cols[col] = col
            continue
        m_q = re.match('^Q\\d+_([1-6])$', col)
        if m_q and current_group:
            new_cols[col] = f'{current_group}_{m_q.group(1)}'
            continue
        m_let = re.match('^([a-z]{2})\\d+_([1-6])$', col)
        if m_let and current_group:
            letters, k = m_let.groups()
            if current_group.startswith(letters):
                new_cols[col] = f'{current_group}_{k}'
                continue
        new_cols[col] = col
    df = df.rename(columns=new_cols)
    keep = [col for col in df.columns if col == 'Q37' or re.match('^[a-z]{2}\\d+_Page Submit$', col) or re.match('^[a-z]{2}\\d+_[1-6]$', col)]

    def sort_key(col: str):
        if col == 'Q37':
            return ('aa', -1, -1)
        m = re.match('^([a-z]{2})(\\d+)_(Page Submit|[1-6])$', col)
        if not m:
            return ('zz', 10 ** 9, 10 ** 9)
        letters, num, tail = m.groups()
        order = 0 if tail == 'Page Submit' else int(tail)
        return (letters, int(num), order)
    return df[sorted(keep, key=sort_key)]

def drop_metadata_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) >= 2:
        return df.iloc[2:].reset_index(drop=True)
    return df.reset_index(drop=True)

def process_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    one_row = gt.iloc[2:3].reset_index(drop=True) if len(gt) >= 3 else gt.iloc[0:1].reset_index(drop=True)
    gt_norm = normalize_columns(one_row)
    drop_cols = [col for col in gt_norm.columns if col.endswith('_Page Submit') or col == 'Q37' or col.startswith('ac1_')]
    return gt_norm.drop(columns=drop_cols)

def filter_ac1_1212(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['ac1_3', 'ac1_4', 'ac1_5', 'ac1_6']
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return df
    vals = df[cols].apply(pd.to_numeric, errors='coerce')
    mask = (vals['ac1_3'] == 1) & (vals['ac1_4'] == 2) & (vals['ac1_5'] == 1) & (vals['ac1_6'] == 2)
    return df.loc[mask].reset_index(drop=True)

def to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce').astype(float)
    s = series.astype(str).str.strip()
    secs_from_timedelta = pd.to_timedelta(s, errors='coerce').dt.total_seconds()
    secs_from_numeric = pd.to_numeric(s, errors='coerce')
    return secs_from_timedelta.where(secs_from_timedelta.notna(), secs_from_numeric).astype(float)

def filter_by_thresholds(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for key, min_sec in thresholds.items():
        col = f'{key}_Page Submit'
        if col in df.columns:
            mask &= to_seconds(df[col]).ge(min_sec).fillna(False)
    return df.loc[mask].reset_index(drop=True)

def load_processed_data(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object], list[str], list[str], list[str]]:
    gt_path = first_existing(data_dir, ['sagat_ground_truth_answers.csv'])
    exp_path = first_existing(data_dir, ['sagat_experimental_responses.csv'])
    con_path = first_existing(data_dir, ['sagat_control_responses.csv'])
    df_exp = read_csv_export(exp_path).dropna()
    df_con = read_csv_export(con_path).dropna()
    gt_raw = read_csv_export(gt_path)
    df_exp = drop_metadata_rows(normalize_columns(df_exp))
    df_con = drop_metadata_rows(normalize_columns(df_con))
    gt_df = process_ground_truth(gt_raw)
    df_exp = filter_by_thresholds(filter_ac1_1212(df_exp), THRESHOLDS)
    df_con = filter_by_thresholds(filter_ac1_1212(df_con), THRESHOLDS)
    pooled = pd.concat([df_con, df_exp], ignore_index=True)
    gt = gt_df.iloc[0].to_dict()
    materials_pos = select_groups_by_valence(gt_df, 'positive')
    materials_neg = select_groups_by_valence(gt_df, 'negative')
    materials_all = materials_pos + materials_neg
    return (pooled, gt, materials_all, materials_pos, materials_neg)

def equal_answer(a: object, b: object) -> bool:
    a_num = pd.to_numeric(pd.Series([a]), errors='coerce').iloc[0]
    b_num = pd.to_numeric(pd.Series([b]), errors='coerce').iloc[0]
    if pd.notna(a_num) and pd.notna(b_num):
        return bool(a_num == b_num)
    if pd.isna(a) or pd.isna(b):
        return False
    return str(a).strip().casefold() == str(b).strip().casefold()

def infer_groups_from_gt(gt_df: pd.DataFrame) -> list[str]:
    groups = {m.group(1) for col in gt_df.columns for m in [re.match('^([a-z]{2}\\d+)_([1-6])$', col)] if m}
    return sorted(groups)

def select_groups_by_valence(gt_df: pd.DataFrame, valence: str) -> list[str]:
    second_letter = {'positive': 'p', 'negative': 'n'}[valence.lower()]
    return [g for g in infer_groups_from_gt(gt_df) if len(g) >= 2 and g[1] == second_letter]

def build_binary_matrix(df: pd.DataFrame, gt: dict[str, object], materials: list[str], question_indices: list[int]) -> pd.DataFrame:
    item_keys = [(material, q) for material in materials for q in question_indices]
    col_names = [f'{material}_Q{q}' for material, q in item_keys]
    rows = []
    for _, row in df.iterrows():
        scored = []
        keep_row = True
        for material, q in item_keys:
            col = f'{material}_{q}'
            if col not in gt or col not in row or pd.isna(row[col]):
                keep_row = False
                break
            scored.append(int(equal_answer(row[col], gt[col])))
        if keep_row:
            rows.append(scored)
    return pd.DataFrame(rows, columns=col_names)

def classical_alpha(items: pd.DataFrame) -> tuple[float, int, int]:
    x = items.dropna(how='any')
    n, k = x.shape
    if n < 2 or k < 2:
        return (np.nan, n, k)
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return (np.nan, n, k)
    alpha = k / (k - 1) * (1 - item_vars.sum() / total_var)
    return (float(alpha), n, k)

def tetrachoric_corr(x: np.ndarray, y: np.ndarray, continuity: float=0.5) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    table = np.zeros((2, 2), dtype=float)
    for xi, yi in zip(x, y):
        table[xi, yi] += 1.0
    if np.any(table == 0):
        table += continuity
    total = table.sum()
    px0 = (table[0, 0] + table[0, 1]) / total
    py0 = (table[0, 0] + table[1, 0]) / total
    p00 = table[0, 0] / total
    ax = norm.ppf(px0)
    ay = norm.ppf(py0)

    def f(rho: float) -> float:
        cov = [[1.0, rho], [rho, 1.0]]
        return float(multivariate_normal.cdf([ax, ay], mean=[0.0, 0.0], cov=cov) - p00)
    lo, hi = (-0.999, 0.999)
    flo, fhi = (f(lo), f(hi))
    if np.isclose(flo, 0.0, atol=1e-08):
        return lo
    if np.isclose(fhi, 0.0, atol=1e-08):
        return hi
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    return float(brentq(f, lo, hi, xtol=1e-06, rtol=1e-06, maxiter=100))

def tetrachoric_corr_matrix(items: pd.DataFrame) -> np.ndarray:
    x = items.to_numpy(dtype=int)
    k = x.shape[1]
    r = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            rij = tetrachoric_corr(x[:, i], x[:, j])
            r[i, j] = r[j, i] = rij
    return r

def alpha_from_corr_matrix(r: np.ndarray) -> float:
    k = r.shape[0]
    if k < 2:
        return np.nan
    total = np.sum(r)
    if total == 0:
        return np.nan
    return float(k / (k - 1) * (1 - k / total))

def ordinal_alpha(items: pd.DataFrame) -> tuple[float, int, int]:
    x = items.dropna(how='any')
    nonzero_var_cols = x.columns[x.nunique(dropna=True) > 1]
    x = x.loc[:, nonzero_var_cols]
    n, k = x.shape
    if n < 2 or k < 2:
        return (np.nan, n, k)
    r = tetrachoric_corr_matrix(x)
    alpha = alpha_from_corr_matrix(r)
    return (alpha, n, k)

def fmt3(x: float) -> str:
    return 'NA' if pd.isna(x) else f'{x:.3f}'

def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def make_row(row: list[str]) -> str:
        return '| ' + ' | '.join((cell.ljust(w) for cell, w in zip(row, widths))) + ' |'
    out = [make_row(headers), '| ' + ' | '.join(('-' * w for w in widths)) + ' |']
    out.extend((make_row(row) for row in str_rows))
    return '\n'.join(out)

def make_table_s8(pooled: pd.DataFrame, gt: dict[str, object], materials_all: list[str]) -> str:
    scales = [('Full perception scale (Q1–Q4)', [1, 2, 3, 4]), ('AV-state subscale (Q2, Q4)', [2, 4]), ('World-state subscale (Q1, Q3)', [1, 3])]
    rows = []
    for scale_name, qs in scales:
        mat = build_binary_matrix(pooled, gt, materials_all, qs)
        classical, n, k_classical = classical_alpha(mat)
        ord_alpha, _, k_ordinal = ordinal_alpha(mat)
        rows.append([scale_name, n, k_classical, k_ordinal, fmt3(classical), fmt3(ord_alpha)])
    return markdown_table(['Scale', 'n', 'Items for classical α', 'Items for ordinal α', 'Classical α', 'Ordinal α'], rows)

def make_table_s9(pooled: pd.DataFrame, gt: dict[str, object], materials_all: list[str]) -> str:
    rows = []
    question_types = {1: 'World-state', 2: 'AV-state', 3: 'World-state', 4: 'AV-state'}
    for q in [1, 2, 3, 4]:
        mat = build_binary_matrix(pooled, gt, materials_all, [q])
        pooled_mean = mat.to_numpy().mean()
        mean_item_var = mat.var(axis=0, ddof=1).mean()
        item_means = mat.mean(axis=0)
        near_ceiling = int((item_means >= CEILING_THRESHOLD).sum())
        rows.append([f'Q{q}', question_types[q], fmt3(pooled_mean), fmt3(mean_item_var), f'{near_ceiling}/{mat.shape[1]}'])
    return markdown_table(['Question', 'Item type', 'Pooled mean accuracy', 'Mean item variance', 'Near-ceiling items'], rows)

def main() -> None:
    pooled, gt, materials_all, _, _ = load_processed_data(DATA_DIR)
    table_s8 = make_table_s8(pooled, gt, materials_all)
    table_s9 = make_table_s9(pooled, gt, materials_all)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(f'Table S8\n{table_s8}\n\nTable S9\n{table_s9}\n', encoding='utf-8')
    print(f'Saved markdown tables to {LOG_FILE}')
if __name__ == '__main__':
    main()
