from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
SCENARIOS = ['CLOSE', 'ASV', 'BIKE']
CLASSES = [1, 2]

def read_results(path: Path) -> List[Tuple[str, Dict[str, List[int]]]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    rows: List[Tuple[str, Dict[str, List[int]]]] = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in SCENARIOS if col not in fieldnames]
        if missing:
            raise ValueError(f'{path} is missing required columns: {missing}')
        id_columns = [col for col in ('participant_id', 'participant', 'id', 'subject_id', 'prolific_id', 'name') if col in fieldnames]
        id_column = id_columns[0] if id_columns else None
        for row_num, row in enumerate(reader, start=2):
            participant_id = (row.get(id_column) or '').strip() if id_column else ''
            if not participant_id:
                participant_id = f'row_{len(rows) + 1:03d}'
            parsed: Dict[str, List[int]] = {}
            for scenario in SCENARIOS:
                raw = (row.get(scenario) or '').strip()
                try:
                    vals = [int(x) for x in raw.split()]
                except ValueError as exc:
                    raise ValueError(f'Could not parse {path}, row {row_num}, column {scenario}: {raw!r}') from exc
                if len(vals) != 3 or any((v not in CLASSES for v in vals)):
                    raise ValueError(f'Expected three labels in {(1, 2)}  at {path}, row {row_num}, column {scenario}; got {raw!r}')
                parsed[scenario] = vals
            rows.append((participant_id, parsed))
    return rows

def align_results(*datasets: List[Tuple[str, Dict[str, List[int]]]]) -> Tuple[List[str], List[Dict[str, Dict[str, List[int]]]]]:
    id_sets = [set(participant_id for participant_id, _ in dataset) for dataset in datasets]
    shared_ids = set.intersection(*id_sets) if id_sets else set()
    all_ids_are_shared = shared_ids and all(len(ids) == len(shared_ids) for ids in id_sets)
    all_ids_are_unique = all(len(ids) == len(dataset) for ids, dataset in zip(id_sets, datasets))
    if all_ids_are_shared and all_ids_are_unique:
        participants = sorted(shared_ids)
        aligned = [{participant_id: values for participant_id, values in dataset} for dataset in datasets]
        return participants, aligned
    lengths = [len(dataset) for dataset in datasets]
    usable_length = min(lengths) if lengths else 0
    participants = [f'row_{i + 1:03d}' for i in range(usable_length)]
    aligned = []
    for dataset in datasets:
        aligned.append({participant: values for participant, (_, values) in zip(participants, dataset[:usable_length])})
    return participants, aligned

def flatten(results: Dict[str, Dict[str, List[int]]], participants: Iterable[str]) -> List[int]:
    vals: List[int] = []
    for participant in participants:
        for scenario in SCENARIOS:
            vals.extend(results[participant][scenario])
    return vals

def agreement(a: List[int], b: List[int]) -> Tuple[int, int, float]:
    if len(a) != len(b):
        raise ValueError('Cannot compare lists with different lengths.')
    n = len(a)
    correct = sum((x == y for x, y in zip(a, b)))
    return (correct, n, correct / n if n else float('nan'))

def cohen_kappa(a: List[int], b: List[int]) -> float:
    correct, n, p_observed = agreement(a, b)
    if n == 0:
        return float('nan')
    p_expected = 0.0
    for c in CLASSES:
        p_a = sum((x == c for x in a)) / n
        p_b = sum((y == c for y in b)) / n
        p_expected += p_a * p_b
    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1.0 - p_expected)

def confusion_counts(a: List[int], b: List[int]) -> Dict[Tuple[int, int], int]:
    counts = {(x, y): 0 for x in CLASSES for y in CLASSES}
    for x, y in zip(a, b):
        counts[x, y] += 1
    return counts

def format_agreement(correct: int, total: int, kappa: float) -> str:
    return f"{correct}/{total} = {correct / total:.1%}; Cohen's kappa = {kappa:.3f}"

def add_confusion_matrix(lines: List[str], title: str, a_name: str, b_name: str, a: List[int], b: List[int]) -> None:
    counts = confusion_counts(a, b)
    lines.append(title)
    lines.append(f'  rows = {a_name}, columns = {b_name}')
    lines.append('             col=1  col=2')
    lines.append(f'  row=1      {counts[1, 1]:>5}  {counts[1, 2]:>5}')
    lines.append(f'  row=2      {counts[2, 1]:>5}  {counts[2, 2]:>5}')
    lines.append('')

def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce LLM/human interrater validation numbers.')
    parser.add_argument('--data-dir', default='data', type=Path, help='Directory containing input CSVs.')
    parser.add_argument('--logs-dir', default='logs', type=Path, help='Directory for the output log file.')
    parser.add_argument('--output-name', default='llm_human_interrater_validation.txt', help='Name of the output log file.')
    args = parser.parse_args()
    llm_file = args.data_dir / 'non_expert_text_rationale_llm_judgments.csv'
    judge1_file = args.data_dir / 'non_expert_text_rationale_human_judge_rater1.csv'
    judge2_file = args.data_dir / 'non_expert_text_rationale_human_judge_rater2.csv'
    llm_rows = read_results(llm_file)
    judge1_rows = read_results(judge1_file)
    judge2_rows = read_results(judge2_file)
    participants, aligned = align_results(llm_rows, judge1_rows, judge2_rows)
    llm, judge1, judge2 = aligned
    if not participants:
        raise ValueError('No shared participants found across the three input files.')
    llm_vals = flatten(llm, participants)
    j1_vals = flatten(judge1, participants)
    j2_vals = flatten(judge2, participants)
    total_expected = len(participants) * len(SCENARIOS) * 3
    assert len(llm_vals) == len(j1_vals) == len(j2_vals) == total_expected
    hh_correct, hh_total, _ = agreement(j1_vals, j2_vals)
    hh_kappa = cohen_kappa(j1_vals, j2_vals)
    llm_j1_correct, llm_j1_total, _ = agreement(llm_vals, j1_vals)
    llm_j2_correct, llm_j2_total, _ = agreement(llm_vals, j2_vals)
    llm_j1_kappa = cohen_kappa(llm_vals, j1_vals)
    llm_j2_kappa = cohen_kappa(llm_vals, j2_vals)
    agree_idx = [i for i, (x, y) in enumerate(zip(j1_vals, j2_vals)) if x == y]
    disagree_idx = [i for i, (x, y) in enumerate(zip(j1_vals, j2_vals)) if x != y]
    llm_agree = [llm_vals[i] for i in agree_idx]
    human_agree = [j1_vals[i] for i in agree_idx]
    agreed_correct, agreed_total, _ = agreement(llm_agree, human_agree)
    agreed_kappa = cohen_kappa(llm_agree, human_agree)
    llm_disagree = [llm_vals[i] for i in disagree_idx]
    j1_disagree = [j1_vals[i] for i in disagree_idx]
    j2_disagree = [j2_vals[i] for i in disagree_idx]
    dis_j1_correct, dis_total, _ = agreement(llm_disagree, j1_disagree)
    dis_j2_correct, _, _ = agreement(llm_disagree, j2_disagree)
    lines: List[str] = []
    lines.append('LLM / human interrater validation for free-form response judgments')
    lines.append('=' * 72)
    lines.append('')
    lines.append('Input files:')
    lines.append(f'  LLM judgments:      {llm_file}')
    lines.append(f'  Human judge 1:      {judge1_file}  [Rater 1]')
    lines.append(f'  Human judge 2:      {judge2_file}  [Rater 2]')
    lines.append('')
    lines.append(f'Shared participants analysed: {len(participants)}')
    lines.append(f"Scenarios: {', '.join(SCENARIOS)}")
    lines.append(f'Judgments per participant: {len(SCENARIOS)} scenarios × 3 anchors = 9')
    lines.append(f'Total judgments compared: {total_expected}')
    lines.append('')
    lines.append('Main manuscript numbers')
    lines.append('-' * 72)
    lines.append('Human-human agreement: ' + format_agreement(hh_correct, hh_total, hh_kappa) + f'  [rounded: {hh_correct}/{hh_total} = {hh_correct / hh_total:.0%}, kappa = {hh_kappa:.2f}]')
    lines.append('LLM vs human judge 1 (Rater 1): ' + format_agreement(llm_j1_correct, llm_j1_total, llm_j1_kappa) + f'  [rounded: {llm_j1_correct}/{llm_j1_total} = {llm_j1_correct / llm_j1_total:.0%}, kappa = {llm_j1_kappa:.2f}]')
    lines.append('LLM vs human judge 2 (Rater 2): ' + format_agreement(llm_j2_correct, llm_j2_total, llm_j2_kappa) + f'  [rounded: {llm_j2_correct}/{llm_j2_total} = {llm_j2_correct / llm_j2_total:.0%}, kappa = {llm_j2_kappa:.2f}]')
    lines.append('LLM on human-agreement subset: ' + format_agreement(agreed_correct, agreed_total, agreed_kappa) + f'  [rounded: {agreed_correct}/{agreed_total} = {agreed_correct / agreed_total:.0%}, kappa = {agreed_kappa:.2f}]')
    lines.append('')
    lines.append('Human-disagreement diagnostic')
    lines.append('-' * 72)
    lines.append(f'Human-agreement cases:    {len(agree_idx)}')
    lines.append(f'Human-disagreement cases: {len(disagree_idx)}')
    lines.append(f'On disagreement cases, LLM agrees with Rater 1:    {dis_j1_correct}/{dis_total} = {dis_j1_correct / dis_total:.1%}')
    lines.append(f'On disagreement cases, LLM agrees with Rater 2: {dis_j2_correct}/{dis_total} = {dis_j2_correct / dis_total:.1%}')
    lines.append('')
    add_confusion_matrix(lines, 'Confusion matrix: human judge 1 vs human judge 2', 'Rater 1', 'Rater 2', j1_vals, j2_vals)
    add_confusion_matrix(lines, 'Confusion matrix: LLM vs human judge 1', 'LLM', 'Rater 1', llm_vals, j1_vals)
    add_confusion_matrix(lines, 'Confusion matrix: LLM vs human judge 2', 'LLM', 'Rater 2', llm_vals, j2_vals)
    add_confusion_matrix(lines, 'Confusion matrix: LLM vs agreed human label', 'LLM', 'Human-agreed', llm_agree, human_agree)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.logs_dir / args.output_name
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Saved log to {output_path}')
if __name__ == '__main__':
    main()
