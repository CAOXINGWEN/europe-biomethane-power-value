"""Regenerate all four 8-panel main figures, ten Extended Data figures, and Supplementary Fig. S1."""
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
for script in ('main_figures.py', 'extended_data_figures.py'):
    print(f'==> Running {script} ...')
    subprocess.run([sys.executable, str(S / script)], check=True)
print('All 15 figures regenerated successfully in ../figures/ (with geometry QA reports in ../qa/).')
