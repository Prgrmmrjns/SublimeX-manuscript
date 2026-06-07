#!/usr/bin/env python3
"""Reproduce manuscript results. From repo root: python scripts/run_all.py [--only STEP]"""
import argparse
import subprocess
import sys
from pathlib import Path

from config import REPO, SCRIPTS

STEPS = {
    'preprocess': ['preprocess.py'],
    'eval': ['main_eval.py'],
    'ablation': ['ablation_study.py'],
    'tables': ['generate_tables.py'],
    'figures': [
        'flowchart.py',
        'feature_analysis.py',
        'domain_interpretation.py',
        'incremental_features.py',
    ],
}


def run(script):
    print(f'\n>>> {script}', flush=True)
    subprocess.check_call([sys.executable, str(SCRIPTS / script)], cwd=SCRIPTS)


def main():
    p = argparse.ArgumentParser(description='SublimeX manuscript reproduction pipeline')
    p.add_argument(
        '--only',
        choices=[*STEPS, 'all'],
        default='all',
        help='Run one stage or the full pipeline (default: all)',
    )
    args = p.parse_args()
    order = STEPS if args.only == 'all' else {args.only: STEPS[args.only]}
    print(f'Repo: {REPO}', flush=True)
    for scripts in order.values():
        for s in scripts:
            run(s)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
