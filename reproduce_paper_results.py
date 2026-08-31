"""Run all scripts that reproduce the paper's figures and statistics.

Each script in SCRIPT_ORDER is run in a subprocess from the repository root;
figures land in plots/, statistics logs in logs/, and each script's full
stdout/stderr in logs/runner/<script>.log. See the README for the
script-to-figure mapping. Total runtime is well under a minute.

Usage: python reproduce_paper_results.py [--only ...] [--skip ...]
       [--continue-on-error] [--dry-run]
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / 'scripts'
PLOTS_DIR = REPO_ROOT / 'plots'
LOGS_DIR = REPO_ROOT / 'logs'
RUNNER_LOG_DIR = LOGS_DIR / 'runner'
SCRIPT_ORDER = ['reproduce_figure3.py', 'reproduce_figureED3.py', 'reproduce_figureED4.py', 'reproduce_figureED5_analysis.py', 'reproduce_figureED7_analysis.py', 'reproduce_SI_LLM_Judge.py', 'reproduce_SI_cronbach_alpha.py']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run all scripts needed to reproduce the paper results.')
    parser.add_argument('--only', nargs='+', choices=SCRIPT_ORDER, help='Run only these script filenames, in the canonical order.')
    parser.add_argument('--skip', nargs='+', choices=SCRIPT_ORDER, default=[], help='Skip these script filenames.')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue running later scripts if one script fails.')
    parser.add_argument('--dry-run', action='store_true', help='Print the scripts that would run, but do not execute them.')
    return parser.parse_args()

def selected_scripts(args: argparse.Namespace) -> list[str]:
    scripts = args.only if args.only is not None else SCRIPT_ORDER
    return [script for script in scripts if script not in set(args.skip)]

def ensure_layout() -> None:
    missing = []
    if not SCRIPTS_DIR.exists():
        missing.append(str(SCRIPTS_DIR))
    if not (REPO_ROOT / 'data').exists():
        missing.append(str(REPO_ROOT / 'data'))
    if missing:
        raise FileNotFoundError('Missing required repo directories:\n  ' + '\n  '.join(missing))
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNNER_LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_script(script_name: str) -> int:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f'Missing script: {script_path}')
    cmd = [sys.executable, str(script_path)]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    # Pin the non-interactive backend so figure PDFs are identical across
    # machines regardless of which GUI backends are installed.
    env['MPLBACKEND'] = 'Agg'
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    combined_output = []
    combined_output.append(f"Command: {' '.join(cmd)}")
    combined_output.append(f'Working directory: {REPO_ROOT}')
    combined_output.append(f'Exit code: {result.returncode}')
    combined_output.append(f'Elapsed seconds: {elapsed:.2f}')
    combined_output.append('\n--- stdout ---')
    combined_output.append(result.stdout.rstrip() or '[empty]')
    combined_output.append('\n--- stderr ---')
    combined_output.append(result.stderr.rstrip() or '[empty]')
    combined_output.append('')
    runner_log = RUNNER_LOG_DIR / f'{Path(script_name).stem}.log'
    runner_log.write_text('\n'.join(combined_output), encoding='utf-8')
    status = 'OK' if result.returncode == 0 else 'FAILED'
    print(f'[{status}] {script_name}  ({elapsed:.1f}s)  log: {runner_log}')
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f'    {line}')
    return result.returncode

def main() -> None:
    args = parse_args()
    ensure_layout()
    scripts = selected_scripts(args)
    if args.dry_run:
        print('Scripts that would run:')
        for script in scripts:
            print(f'  - {script}')
        return
    failures: list[str] = []
    print(f'Running {len(scripts)} reproduction scripts from {SCRIPTS_DIR}')
    print(f'Plots directory: {PLOTS_DIR}')
    print(f'Logs directory:  {LOGS_DIR}\n')
    for script in scripts:
        exit_code = run_script(script)
        if exit_code != 0:
            failures.append(script)
            if not args.continue_on_error:
                break
    if failures:
        print('\nFailed scripts:')
        for script in failures:
            print(f'  - {script}')
        print(f'See detailed logs in {RUNNER_LOG_DIR}')
        raise SystemExit(1)
    print('\nAll requested paper results were reproduced successfully.')
    print(f'Plots saved under: {PLOTS_DIR}')
    print(f'Logs saved under:  {LOGS_DIR}')
if __name__ == '__main__':
    main()
