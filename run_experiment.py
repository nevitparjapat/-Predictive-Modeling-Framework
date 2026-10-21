"""Re-run the whole experiment from the report in the terminal.

    python run_experiment.py

Prints the key numbers and saves the simulated dataset to data/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cropcast import pipeline
from cropcast.data import make_dataset


def main():
    df = make_dataset(seed=42)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/synthetic_yield_data.csv", index=False)
    res = pipeline.run(df)
    print(f"Rows: {len(df)}  |  split: {res.split_sizes}")
    print(f"\n{'Model':34s}{'RMSE':>8s}{'MAE':>8s}{'MAPE%':>8s}{'R2':>8s}{'Bias':>8s}")
    for name, m in res.metrics.items():
        print(f"{name:34s}{m['RMSE']:8.3f}{m['MAE']:8.3f}{m['MAPE']:8.1f}{m['R2']:8.3f}{m['Bias']:8.3f}")
    print("\nPass marks for Gradient Boosting:")
    for row in res.card:
        print(f"  [{'PASS' if row['passed'] else 'FAIL'}] {row['check']:42s} {row['result']:>10s}  (needs {row['required']})")


if __name__ == "__main__":
    main()
