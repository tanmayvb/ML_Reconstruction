import pandas as pd
from pathlib import Path

#==========================================
#Run after all 81 jobs finish:
#uv run python scripts/build_summary.py; which creates:
#summary.csv
#containing all: 9 trainings × 9 evaluations = 81 rows
#==========================================


all_results = []

for csv_file in Path("experiments").rglob("results.csv"):

    df = pd.read_csv(csv_file)

    all_results.append(df)

summary = pd.concat(
    all_results,
    ignore_index=True
)

summary.sort_values(
    "f1",
    ascending=False
).to_csv(
    "summary.csv",
    index=False
)

print(
    summary[
        [
            "lr",
            "patches_per_epoch",
            "threshold",
            "min_size",
            "precision",
            "recall",
            "f1"
        ]
    ].head(20)
)

best = summary.sort_values(
    "f1",
    ascending=False
).iloc[0]
