"""Run every stage in order. First run needs network; later runs are offline."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for stage in sorted(HERE.glob("0*_*.py")):
    print(f"\n=== {stage.name} ===")
    subprocess.run([sys.executable, str(stage)], check=True)
