"""Reproduce the Extended Data belief-update ("arrows") figure.

(Cited as Extended Data Fig. 3 in the paper body text; the corresponding
caption in the current proof is Extended Data Fig. 4 - see the README.)

For each participant, draws an arrow from their average signed confidence
before the explanation to after the explanation, for the nearest-neighbour
task (panels a, b) and the prediction task (panels c, d), split by experts
(a, c) and non-experts (b, d). Signed confidence is positive when the answer
is correct and negative when it is wrong.

Inputs: data/expert_simulator_responses.csv and
data/non_expert_simulator_responses.csv.
Output: plots/combined_movement_2x2.pdf.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
FS_AXIS_LABEL = 11
FS_TICK_LABEL = 9
FS_LEGEND = 11
FS_TITLE_TEXT = 13
FS_PANEL_LABEL = 14
FS_MENTAL_MODEL = 9
FS_ROW_LABEL = 13
QUESTIONS = ('a', 'b', 'c')  # survey question blocks: a=ASV, b=BIKE, c=CLOSE
GT_OWN_ANSWER = {'a': 2, 'b': 2, 'c': 2}  # correct nearest-neighbour answer per block
GT_PREDICTION = {'a': 2, 'b': 2, 'c': 1}  # correct prediction answer per block
DATA_FILES = {'Experts': 'expert_simulator_responses.csv', 'Non-Experts': 'non_expert_simulator_responses.csv'}
COLORS = {'improved': '#2ca02c', 'worsened': '#d62728', 'unchanged': 'gray'}

def required_columns() -> set[str]:
    suffixes = ('1', '2_1', '3', '4_1', '6', '7_1', '8', '9_1')
    return {f'{question}{suffix}' for question in QUESTIONS for suffix in suffixes}

def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for label, filename in DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing required input file: {path}')
        datasets[label] = pd.read_csv(path)
    validate_columns(datasets)
    return datasets

def validate_columns(datasets: Dict[str, pd.DataFrame]) -> None:
    needed = required_columns()
    missing_messages = []
    for label, df in datasets.items():
        missing = sorted(needed.difference(df.columns))
        if missing:
            missing_messages.append(f"{label}: {', '.join(missing)}")
    if missing_messages:
        details = '\n'.join(missing_messages)
        raise ValueError(f'Missing required columns:\n{details}')

def get_spectrum_score(answer: object, confidence: object, correct_answer: int) -> Optional[float]:
    try:
        answer_value = int(answer)
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return None
    if np.isnan(confidence_value):
        return None
    return confidence_value if answer_value == correct_answer else -confidence_value

def plot_movement(ax: plt.Axes, df: pd.DataFrame, title: str, task: str, show_ylabel: bool, panel_label: str) -> None:
    if task == 'own_answer':
        ground_truth = GT_OWN_ANSWER
        start_answer, start_conf = ('1', '2_1')
        end_answer, end_conf = ('6', '7_1')
        xlabel = 'Average confidence score'
    elif task == 'prediction':
        ground_truth = GT_PREDICTION
        start_answer, start_conf = ('3', '4_1')
        end_answer, end_conf = ('8', '9_1')
        xlabel = 'Average prediction confidence score'
    else:
        raise ValueError(f'Unknown task: {task}')
    add_background(ax)
    for participant_idx, row in df.iterrows():
        start_scores = []
        end_scores = []
        for question in QUESTIONS:
            correct_answer = ground_truth[question]
            start_score = get_spectrum_score(row[f'{question}{start_answer}'], row[f'{question}{start_conf}'], correct_answer)
            end_score = get_spectrum_score(row[f'{question}{end_answer}'], row[f'{question}{end_conf}'], correct_answer)
            if start_score is not None:
                start_scores.append(start_score)
            if end_score is not None:
                end_scores.append(end_score)
        if not start_scores or not end_scores:
            continue
        start_avg = float(np.mean(start_scores))
        end_avg = float(np.mean(end_scores))
        draw_participant_arrow(ax, start_avg, end_avg, participant_idx)
    format_axis(ax, df, title, xlabel, show_ylabel, panel_label)

def add_background(ax: plt.Axes) -> None:
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvspan(-8, 0, color='#ffe6e6', alpha=0.2)
    ax.axvspan(0, 8, color='#e6ffe6', alpha=0.2)

def draw_participant_arrow(ax: plt.Axes, start_avg: float, end_avg: float, y_position: int) -> None:
    delta = end_avg - start_avg
    if delta > 0:
        color = COLORS['improved']
    elif delta < 0:
        color = COLORS['worsened']
    else:
        color = COLORS['unchanged']
    if start_avg == end_avg:
        ax.scatter(start_avg, y_position, color=color, s=45, alpha=0.5)
        return
    ax.annotate('', xy=(end_avg, y_position), xytext=(start_avg, y_position), arrowprops={'arrowstyle': '->', 'color': color, 'lw': 2.0, 'shrinkA': 0, 'shrinkB': 0})
    ax.scatter(start_avg, y_position, color=color, s=25, alpha=0.6)

def format_axis(ax: plt.Axes, df: pd.DataFrame, title: str, xlabel: str, show_ylabel: bool, panel_label: str) -> None:
    x_ticks = [-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]
    x_labels = ['7', '6', '5', '4', '3', '2', '1', '1', '2', '3', '4', '5', '6', '7']
    ax.set_title(title, fontsize=FS_TITLE_TEXT, pad=16)
    ax.set_xlim(-8, 8)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=FS_TICK_LABEL)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL)
    if show_ylabel:
        ax.set_ylabel('Participant number', fontsize=FS_AXIS_LABEL)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f'P{i + 1}' for i in range(len(df))], fontsize=FS_TICK_LABEL)
    ax.set_ylim(-1, len(df))
    y_top = len(df)
    ax.text(-4, y_top, 'Wrong mental model', color=COLORS['worsened'], fontsize=FS_MENTAL_MODEL, ha='center', va='bottom', fontweight='bold')
    ax.text(4, y_top, 'Correct mental model', color=COLORS['improved'], fontsize=FS_MENTAL_MODEL, ha='center', va='bottom', fontweight='bold')
    ax.text(-0.2, 1.02, panel_label, transform=ax.transAxes, fontsize=FS_PANEL_LABEL, fontweight='bold', va='bottom', ha='left')

def add_row_labels(fig: plt.Figure) -> None:
    fig.text(0.005, 0.745, 'Nearest Neighbor task', rotation=90, fontsize=FS_ROW_LABEL, fontweight='bold', va='center', ha='left')
    fig.text(0.005, 0.305, 'Prediction task', rotation=90, fontsize=FS_ROW_LABEL, fontweight='bold', va='center', ha='left')

def add_legend(fig: plt.Figure) -> None:
    handles = [mpatches.Patch(color=COLORS['improved'], label='Improved'), mpatches.Patch(color=COLORS['worsened'], label='Worsened'), mpatches.Patch(color=COLORS['unchanged'], label='No change')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.002), ncol=3, fontsize=FS_LEGEND, frameon=False)

def make_figure(datasets: Dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.0))
    plot_movement(axes[0, 0], datasets['Experts'], title='Experts', task='own_answer', show_ylabel=True, panel_label='a')
    plot_movement(axes[0, 1], datasets['Non-Experts'], title='Non-Experts', task='own_answer', show_ylabel=False, panel_label='b')
    plot_movement(axes[1, 0], datasets['Experts'], title='Experts', task='prediction', show_ylabel=True, panel_label='c')
    plot_movement(axes[1, 1], datasets['Non-Experts'], title='Non-Experts', task='prediction', show_ylabel=False, panel_label='d')
    add_row_labels(fig)
    add_legend(fig)
    plt.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.09, wspace=0.25, hspace=0.42)
    return fig

def save_figure(fig: plt.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'combined_movement_2x2.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved {output_path}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot mental-model movement for experts and non-experts.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Directory containing expert_simulator_responses.csv and non_expert_simulator_responses.csv. Default: data/')
    parser.add_argument('--output-dir', type=Path, default=Path('plots'), help='Directory where plots are saved. Default: plots/')
    parser.add_argument('--show', action='store_true', help='Display the plot interactively after saving.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    datasets = load_data(args.data_dir)
    fig = make_figure(datasets)
    save_figure(fig, args.output_dir)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
if __name__ == '__main__':
    main()
