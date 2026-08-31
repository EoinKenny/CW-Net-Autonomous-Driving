"""Reproduce Figure 3 (on-road private-track results).

Panels and outputs (saved to plots/):
  Fig 3a (CLOSE): CLOSE_global.pdf (speed vs Close-concept probability with OLS
    fit; Pearson r and intercept printed to stdout) and CLOSE_local.pdf
    (Close-concept probability over time with driving-mode shading).
  Fig 3b (ASV):   ASV_global.pdf (speed/ASV traces around ASV activations) and
    ASV_local_asv.pdf / ASV_local_speed.pdf (Dynamic Time Warping alignment of
    the cone-intervention trials; DTW distances printed to stdout).
  Fig 3c (BIKE):  BIKE_global.pdf (speed around manual-to-autonomous
    transitions) and BIKE_local.pdf (engagement speeds before vs after the
    explanation, by cyclist behaviour).

Inputs: the figure3_*_concept_probabilities.csv / figure3_*_vehicle_modes.csv
pairs in data/ (anonymised vehicle logs). Run from the repository root, or via
reproduce_paper_results.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
try:
    from fastdtw import fastdtw
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'fastdtw is required to reproduce the Fig 3b DTW panels '
        '(ASV_local_asv.pdf, ASV_local_speed.pdf). '
        "Install it with: pip install 'fastdtw>=0.3'"
    ) from exc
DATA = Path('data')
PLOTS = Path('plots')
PLOTS.mkdir(exist_ok=True)
CLOSE_CONCEPT_FILE = 'figure3_close_concept_probabilities.csv'
CLOSE_MODES_FILE = 'figure3_close_vehicle_modes.csv'
BIKE_BEFORE_CONCEPT_FILE = 'figure3_bike_before_explanation_concept_probabilities.csv'
BIKE_BEFORE_MODES_FILE = 'figure3_bike_before_explanation_vehicle_modes.csv'
BIKE_AFTER_CONCEPT_FILE = 'figure3_bike_after_explanation_concept_probabilities.csv'
BIKE_AFTER_MODES_FILE = 'figure3_bike_after_explanation_vehicle_modes.csv'
ASV_CONCEPT_FILE = 'figure3_asv_cone_intervention_concept_probabilities.csv'
ASV_MODES_FILE = 'figure3_asv_cone_intervention_vehicle_modes.csv'

def rolling_average(x, n=30):
    """Trailing moving average over the last n samples (the logs are ~10 Hz)."""
    return np.array([np.mean(x[max(0, i - n + 1):i + 1]) for i in range(len(x))])

def parse_clock_minutes(values):
    """Parse Excel-mangled 'MM:SS.s' timestamps into seconds within the hour.

    figure3_close_vehicle_modes.csv was round-tripped through Excel, which
    reduced its ISO timestamps to minute:second strings (e.g. '49:10.3'),
    losing the date and hour. This recovers a monotonically increasing seconds
    counter by unwrapping the hour rollovers.
    """
    s = values.astype(str).str.strip()
    parts = s.str.extract(r'^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$')
    if parts.isna().any().any():
        return None
    seconds_in_hour = parts[0].astype(float) * 60 + parts[1].astype(float)
    unwrapped = []
    offset = 0.0
    previous = None
    for value in seconds_in_hour:
        if previous is not None and value + offset < previous - 1800:
            offset += 3600.0
        unwrapped_value = value + offset
        unwrapped.append(unwrapped_value)
        previous = unwrapped_value
    return pd.Series(unwrapped, index=values.index, dtype=float)

def parse_timestamps(series, reference=None):
    """Parse ISO timestamps, falling back to parse_clock_minutes for the
    Excel-mangled CLOSE modes file (anchored to the concept file's hour)."""
    values = series.astype(str).str.strip()
    parsed = pd.Series(pd.to_datetime(values, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'), index=series.index)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    if parsed.isna().any():
        clock_minutes = parse_clock_minutes(values.loc[parsed.isna()])
        if clock_minutes is not None and reference is not None and len(reference) > 0:
            reference_min = pd.Series(reference).min()
            base = reference_min.replace(minute=0, second=0, microsecond=0, nanosecond=0)
            converted = base + pd.to_timedelta(clock_minutes, unit='s')
            while converted.max() < reference_min - pd.Timedelta(minutes=30):
                converted = converted + pd.Timedelta(hours=1)
            while converted.min() > reference_min + pd.Timedelta(minutes=30):
                converted = converted - pd.Timedelta(hours=1)
            parsed.loc[parsed.isna()] = converted
    if parsed.isna().any():
        bad_values = values.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f'Could not parse timestamp values: {bad_values}')
    return parsed

def load_df(concept_filename, modes_filename, concept_columns=None):
    """Load a concept-probability CSV, join the nearest vehicle-mode row by
    timestamp, and add a 'clock' column (seconds since the first sample)."""
    df = pd.read_csv(DATA / concept_filename)
    if concept_columns is not None:
        if len(df.columns) != len(concept_columns):
            raise ValueError(f'{concept_filename} has {len(df.columns)} columns, expected {len(concept_columns)}')
        df.columns = concept_columns
    modes = pd.read_csv(DATA / modes_filename)
    if modes['autonomous'].dtype != bool:
        # Guard against CSV corruption (e.g. an Excel round-trip turning the
        # column into 'TRUE'/'FALSE' strings, which would silently misparse).
        raise TypeError(f"{modes_filename}: 'autonomous' column parsed as {modes['autonomous'].dtype}, expected bool")
    df['timestamp'] = parse_timestamps(df['timestamp'])
    modes['timestamp'] = parse_timestamps(modes['timestamp'], reference=df['timestamp'])
    df, modes = (df.sort_values('timestamp'), modes.sort_values('timestamp'))
    df = pd.merge_asof(df, modes, on='timestamp', direction='nearest')
    df['clock'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    return df

def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS / name, dpi=300, bbox_inches='tight')
    plt.close()

def reproduce_close():
    """Fig 3a: relationship between speed and the Close concept in the CLOSE
    scenario, plus the concept trace around the stuck-next-to-parked-cars event."""
    concept = 'Close'
    # The CLOSE concept CSV has no header row; these are its columns.
    close_columns = ['timestamp', 'Left', 'Right', 'Straight', 'Stop', 'Slow', 'ASV', 'Intersection', 'Close']
    df = load_df(CLOSE_CONCEPT_FILE, CLOSE_MODES_FILE, concept_columns=close_columns)
    # Restrict to the CLOSE test drive (30 s - 1487 s), trimming the vehicle
    # start-up and post-test log segments.
    df = df[(df.clock > 30) & (df.clock < 1487)].copy()
    x = df[df['autonomous']]['speed']
    y = df[df['autonomous']][concept]
    coefficients = np.polyfit(x, y, 1)
    p = np.poly1d(coefficients)
    _, intercept = coefficients
    r, p_value = pearsonr(x, y)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=x, y=y, s=100, color='#1f77b4', edgecolor='w', alpha=0.8)
    plt.plot(x, p(x), color='#ff7f0e', linewidth=3, label='Ordinary Least Squares')
    plt.xlabel('Car Speed (m/s)', fontsize=16)
    plt.ylabel('Close Concept Probability', fontsize=16)
    plt.text(0.95, 0.05, f'Pearson r: {r:.3f}\nIntercept: {intercept:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.5'))
    plt.axhline(y=intercept, color='#d62728', linestyle='--', linewidth=1.5, label='Intercept')
    plt.legend(loc='upper right', fontsize=14, frameon=True)
    savefig('CLOSE_global.pdf')
    df[concept] = rolling_average(df[concept].values, n=30)
    # Zoom on the 100 s - 260 s window containing the stuck event shown in Fig 3a.
    subset = df[(df.clock > 100) & (df.clock < 260)].copy()
    x_values = subset['clock'] - subset['clock'].min()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(x_values, subset[concept], linewidth=3, color='#1f77b4', label='Close Concept Probability')
    ax.axhline(y=intercept, color='#d62728', linestyle='--', linewidth=2, label='Intercept')
    y_min, y_max = ax.get_ylim()
    auto = subset['autonomous'].astype(bool).values
    ax.fill_between(x_values, y_min, y_max, where=auto, color='#2ca02c', alpha=0.2, label='Self-Driving Mode')
    ax.fill_between(x_values, y_min, y_max, where=~auto, color='#d62728', alpha=0.2, label='Manual Mode')
    ax.set_xlabel('Time (sec)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Close Concept Probability', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.title('Close Concept Probability Over Time', fontsize=21, fontweight='bold', pad=20)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=14, framealpha=0.8, edgecolor='none')
    savefig('CLOSE_local.pdf')
    print(f'CLOSE: r={r:.4f}, p={p_value:.4g}, coefficients={coefficients}')
    print('Saved CLOSE_global.pdf and CLOSE_local.pdf')

def to_seconds(mmss, offset):
    """Convert a manually annotated 'MM:SS' video time to log-clock seconds.
    `offset` is the video-to-log clock difference for that recording."""
    mins, secs = map(int, mmss.split(':'))
    return mins * 60 + secs - offset

def slice_rows(df, rows, offset):
    """Slice the log into one DataFrame per annotated (start, end) event window."""
    out = []
    for start, end in rows:
        a, b = (to_seconds(start, offset), to_seconds(end, offset))
        out.append(df[(df.clock > a) & (df.clock < b)])
    return out

def speed_at_auto_or_max(temp):
    """Speed at the moment self-driving was engaged within an event window
    (or the maximum speed if the driver never engaged it)."""
    auto = temp['autonomous'].astype(bool).values
    speed = temp['speed'].values
    idx = list(auto).index(True) if True in auto else np.argmax(speed)
    return speed[idx]

def collect_transition_data(df, hz=10, after_seconds=5):
    """Collect speed around every manual-to-autonomous transition: mean speed
    in the second before, speed at the transition, and per-second means for
    `after_seconds` seconds after (logs are sampled at `hz` Hz)."""
    before, transition, after = ([], [], [])
    auto = df['autonomous'].astype(bool).values
    speed = df['speed'].values
    for i in range(1, len(df)):
        if not auto[i - 1] and auto[i]:
            before.append(np.mean(speed[max(0, i - hz):i]))
            transition.append(speed[i])
            after.append([np.mean(speed[i + hz * j:i + hz * (j + 1)]) if i + hz * j < len(speed) else np.nan for j in range(after_seconds)])
    return (before, transition, after)

def plot_bike_global(before, transition, after):
    """Fig 3c global panel: speed trajectories around every manual-to-autonomous
    transition in the before-explanation cyclist tests."""
    x = np.arange(-1, 6)
    plt.figure(figsize=(5, 5))
    sns.set_style('whitegrid')
    for b, t, a in zip(before, transition, after):
        plt.plot(x, [b, t] + a, alpha=0.5, linewidth=1.5, color='#1f77b4')
    mean_speeds = [np.mean(before), np.mean(transition)] + list(np.nanmean(after, axis=0))
    plt.plot(x, mean_speeds, color='#d62728', linewidth=3, label='Mean Speed')
    plt.axvspan(-1.5, 0, facecolor='#ff9999', alpha=0.4, zorder=-1)
    plt.axvspan(0, 5.5, facecolor='#90ee90', alpha=0.4, zorder=-1)
    plt.xlabel('Seconds', fontsize=14)
    plt.ylabel('Speed (m/s)', fontsize=14)
    plt.title('Speed Transitioning to Self-Driving Mode', fontsize=14)
    plt.xticks(x, ['-1', 'Start', '+1', '+2', '+3', '+4', '+5'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(handles=[Patch(facecolor='#90ee90', alpha=0.4, label='Self-Driving Mode'), Patch(facecolor='#ff9999', alpha=0.4, label='Manual Mode'), plt.Line2D([0], [0], color='#1f77b4', lw=1.5, alpha=0.5, label='Speed Trajectories'), plt.Line2D([0], [0], color='#d62728', lw=3, label='Mean Speed')], fontsize=12, loc='lower left')
    plt.annotate(f'Start: {mean_speeds[1]:.2f} m/s', xy=(0, mean_speeds[1]), xytext=(0, mean_speeds[1] + 1), arrowprops=dict(arrowstyle='->', color='black'), fontsize=15, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    plt.annotate(f'+5s: {mean_speeds[-1]:.2f} m/s', xy=(5, mean_speeds[-1]), xytext=(3, mean_speeds[-1] - 2), arrowprops=dict(arrowstyle='->', color='black'), fontsize=15, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    savefig('BIKE_global.pdf')

def reproduce_bike():
    """Fig 3c: cyclist (BIKE) scenario. 'serial'/'before' = tests before the
    safety driver saw the CW-Net explanation; 'parallel'/'after' = tests after."""
    serial_df = load_df(BIKE_BEFORE_CONCEPT_FILE, BIKE_BEFORE_MODES_FILE)
    parallel_df = load_df(BIKE_AFTER_CONCEPT_FILE, BIKE_AFTER_MODES_FILE)
    before, transition, after = collect_transition_data(serial_df)
    plot_bike_global(before, transition, after)
    # Cyclist-test event windows, hand-annotated as 'MM:SS' video times while
    # reviewing the recordings of the two test drives.
    serial_rows = [('10:11', '10:59'), ('13:10', '13:42'), ('15:20', '15:50'), ('17:18', '17:50'), ('18:53', '19:34'), ('22:20', '22:44'), ('25:07', '25:37'), ('26:39', '27:12'), ('58:40', '59:30'), ('61:23', '62:10'), ('64:18', '64:56'), ('67:27', '67:53'), ('70:40', '71:04'), ('72:38', '73:15'), ('78:36', '79:32'), ('81:23', '82:25'), ('85:25', '85:56'), ('86:38', '87:04'), ('88:07', '88:40'), ('88:32', '89:27')]
    parallel_rows = [('4:40', '5:25'), ('6:59', '7:37'), ('8:55', '9:30'), ('11:43', '12:30'), ('13:20', '14:00'), ('16:50', '17:50'), ('18:35', '19:33'), ('20:27', '21:00'), ('24:17', '25:20'), ('26:02', '26:53')]
    # offset = video-to-log clock difference for each recording (seconds).
    serial_series = slice_rows(serial_df, serial_rows, offset=35)
    parallel_series = slice_rows(parallel_df, parallel_rows, offset=40)
    # Keep only the trials that constitute complete, labelled cyclist tests
    # (the dropped windows were aborted or otherwise unusable trials);
    # serial_labels/parallel_labels below give each kept trial's cyclist
    # behaviour, in the same order.
    serial_indices = [0, 3, 4, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19]
    serial_series = [serial_series[i] for i in serial_indices]
    parallel_series = parallel_series[1:]
    serial_labels = ['Crossing', 'Stationary', 'Stationary', 'Stationary', 'Crossing', 'Crossing', 'Crossing', 'Crossing', 'Swaying', 'Swaying', 'Swaying', 'Swaying', 'Swaying', 'Swaying']
    parallel_labels = ['Stationary', 'Stationary', 'Stationary', 'Stationary', 'Crossing', 'Swaying', 'Swaying', 'Swaying', 'Swaying']
    plot_df = pd.DataFrame({'Speed Before Engaging Self-Driving (m/s)': [speed_at_auto_or_max(x) for x in serial_series] + [speed_at_auto_or_max(x) for x in parallel_series], 'Label': ['Before Explanation'] * len(serial_series) + ['After Explanation'] * len(parallel_series), 'Category': serial_labels + parallel_labels})
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(data=plot_df, x='Category', y='Speed Before Engaging Self-Driving (m/s)', hue='Label', ax=ax)
    savefig('BIKE_local.pdf')
    print('Saved BIKE_global.pdf and BIKE_local.pdf')

def collect_asv_series(df, threshold, reject_high_asv=False):
    """Extract time series around ASV-concept activations.

    Scans for moments where the 1-second-ahead mean ASV probability crosses
    `threshold` while the car is in self-driving mode, and returns up to 15 s
    of [speed, ASV, clock] after each onset (truncated at the first manual
    takeover). With reject_high_asv=True, keeps only sub-0.5 activations that
    occur while moving and away from any 0.5+ spike (the '0.25 ASV' baseline
    series in Fig 3b).
    """
    result, skip = ([], False)
    len_before, len_after = (2, 150)  # samples at 10 Hz: 0.2 s before, 15 s after
    v = df['ASV'].values
    for i in range(len_before, len(v)):
        if skip:
            # Still inside the previous activation; wait until it subsides.
            if v[i:i + 10].mean() > threshold:
                continue
            skip = False
        if v[i:i + 10].mean() <= threshold:
            continue
        asv = df.iloc[i - len_before:i + len_after].copy()
        # Only keep onsets that happen fully in self-driving mode.
        if asv.iloc[len_before - 2:len_before + 2]['autonomous'].mean() != 1.0:
            continue
        if reject_high_asv:
            if asv.iloc[len_before - 2:len_before + 2]['speed'].mean() < 0.05:
                continue
            if (asv['ASV'] > 0.5).mean() > 0:
                continue
            if (df.iloc[max(0, i - 50):i]['ASV'] > 0.5).mean() > 0:
                continue
        # Truncate the series at the first manual takeover, if any.
        counter = 0
        for _, row in asv.iloc[len_before:].iterrows():
            counter += 1
            if not row['autonomous']:
                asv = asv.iloc[len_before:len_before + counter]
                break
        result.append([asv['speed'].values.tolist(), asv['ASV'].values.tolist(), asv['clock'].values.tolist()])
        skip = True
    return result

def pad(data, length):
    """Right-pad each series with NaN so all series have the same length."""
    return np.array([np.pad(x, (0, length - len(x)), constant_values=np.nan) for x in data])

def mean_se(x):
    """Column-wise mean and standard error, ignoring NaN padding."""
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    n = np.sum(~np.isnan(x), axis=0)
    return (mean, std / np.sqrt(n))

def plot_asv_global(result_data1, result_data2):
    """Fig 3b global panel: mean +/- s.e.m. speed and ASV probability after
    0.5-threshold ASV spikes vs 0.25-threshold baseline activations."""
    speeds1, asv1 = ([r[0] for r in result_data1], [r[1] for r in result_data1])
    speeds2, asv2 = ([r[0] for r in result_data2], [r[1] for r in result_data2])
    max_len = max(max(map(len, speeds1)), max(map(len, speeds2)))
    speeds1_mean, speeds1_se = mean_se(pad(speeds1, max_len))
    speeds2_mean, speeds2_se = mean_se(pad(speeds2, max_len))
    asv1_mean, asv1_se = mean_se(pad(asv1, max_len))
    asv2_mean, asv2_se = mean_se(pad(asv2, max_len))
    t = np.arange(max_len) / 10
    plt.figure(figsize=(4, 4))
    plt.plot(t, speeds1_mean, label='Speed m/s: After 0.5 ASV Spike', color='blue')
    plt.fill_between(t, speeds1_mean - speeds1_se, speeds1_mean + speeds1_se, color='blue', alpha=0.2)
    plt.plot(t, speeds2_mean, label='Speed m/s: After 0.25 ASV', color='red')
    plt.fill_between(t, speeds2_mean - speeds2_se, speeds2_mean + speeds2_se, color='red', alpha=0.2)
    plt.plot(t, asv1_mean, label='Probability: ASV 0.5 Spike', color='green')
    plt.fill_between(t, asv1_mean - asv1_se, asv1_mean + asv1_se, color='green', alpha=0.2)
    plt.plot(t, asv2_mean, label='Probability: ASV 0.25', color='orange')
    plt.fill_between(t, asv2_mean - asv2_se, asv2_mean + asv2_se, color='orange', alpha=0.2)
    plt.axhline(y=0.5, color='red', linestyle='--', label='ASV Threshold')
    plt.xlabel('Seconds into series')
    plt.ylabel('Values (Speed m/s and Probability)')
    plt.legend()
    savefig('ASV_global.pdf')

def plot_dtw(stats_data, value_idx, ylabel, filename):
    """Fig 3b DTW panel: align the no-cone vs with-cone trial and report the
    DTW distance (1.61 for ASV probability, 7.37 for speed in the paper)."""
    a = np.asarray(stats_data[0][value_idx], dtype=float)
    b = np.asarray(stats_data[1][value_idx], dtype=float)
    c = np.asarray(stats_data[1][2], dtype=float)
    distance, path = fastdtw(a[:, None], b[:, None], dist=euclidean)
    relative_time = c - c[0]
    plt.figure(figsize=(4, 4))
    for i, j in path:
        if i < len(relative_time) and j < len(relative_time):
            plt.plot([relative_time[i], relative_time[j]], [a[i], b[j]], 'k-', alpha=0.3, linewidth=0.8)
    plt.plot(relative_time[:len(a)], a, label='No Cone', color='#1f77b4', linewidth=2, alpha=0.7)
    plt.plot(relative_time[:len(b)], b, label='With Cone', color='#ff7f0e', linewidth=2, alpha=0.7)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('Dynamic Time Warping')
    plt.legend(loc='best', fontsize=12)
    savefig(filename)
    print(f'{filename}: DTW distance={distance:.2f}')

def reproduce_asv():
    """Fig 3b: ASV (hallucinated stopped vehicle) scenario, including the
    traffic-cone intervention trials."""
    df = load_df(ASV_CONCEPT_FILE, ASV_MODES_FILE)
    df['ASV'] = rolling_average(df['ASV'].values, n=30)
    result_data1 = collect_asv_series(df, threshold=0.5, reject_high_asv=False)
    result_data2 = collect_asv_series(df, threshold=0.25, reject_high_asv=True)
    plot_asv_global(result_data1, result_data2)
    # Cone-intervention trial windows, annotated as video times while
    # reviewing the recording; delay = video-to-log clock difference (seconds).
    delay = 34
    cone_no = df[(df.clock > 3787 - delay) & (df.clock < 3806 - delay)]
    cone_yes = df[(df.clock > 3925 - delay) & (df.clock < 3952 - delay)]
    stats_data = [[cone_no['speed'].values, cone_no['ASV'].values, cone_no['clock'].values], [cone_yes['speed'].values, cone_yes['ASV'].values, cone_yes['clock'].values]]
    plot_dtw(stats_data, value_idx=1, ylabel='ASV Probability', filename='ASV_local_asv.pdf')
    plot_dtw(stats_data, value_idx=0, ylabel='Speed m/s', filename='ASV_local_speed.pdf')
    print('Saved ASV_global.pdf, ASV_local_asv.pdf, and ASV_local_speed.pdf')
if __name__ == '__main__':
    reproduce_asv()
    reproduce_bike()
    reproduce_close()
    print(f'All figures saved in {PLOTS.resolve()}')
