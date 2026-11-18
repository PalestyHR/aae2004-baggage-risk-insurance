"""Run the reported numerical experiments and verify their results."""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    steps = [
        ("Forecasting model", ["scripts/reproduce_risk.py"]),
        ("Baseline models", ["scripts/train_baselines.py"]),
        ("Loss-ratio comparison", ["scripts/compare_loss_ratios.py"]),
        (
            "Insurance optimisation",
            [
                "scripts/reproduce_insurance.py",
                "--forecast",
                "results/reproduced_risk/forecast_interface_2026_08.csv",
            ],
        ),
        (
            "Monte Carlo simulation",
            [
                "scripts/simulate_selected_menu.py",
                "--forecast",
                "results/reproduced_risk/forecast_interface_2026_08.csv",
                "--menu",
                "results/reproduced_insurance/selected_menu.json",
            ],
        ),
        (
            "Baseline comparison table",
            ["scripts/export_baseline_comparison.py", "--reproduced"],
        ),
        ("Numerical verification", ["scripts/check_reproduction.py", "--full"]),
    ]
    started = time.perf_counter()

    for index, (title, arguments) in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {title}", flush=True)
        completed = subprocess.run([sys.executable, *arguments], cwd=ROOT)

        if completed.returncode:
            raise SystemExit(completed.returncode)

    elapsed = time.perf_counter() - started
    print(f"\nReproduction complete: PASS ({elapsed:.1f} seconds)")
    print(f"Verification report: {ROOT / 'results/reproduction_check.json'}")


if __name__ == "__main__":
    main()
