sbatch job.slurm config.yaml


After runs, you get:

results.csv

Example:

percentile,sigma_xy,threshold,precision,recall,f1,n_spines
70,5,0.6,0.40,0.27,0.32,209
60,4,0.5,0.45,0.30,0.36,230


Optional: analyze results
import pandas as pd

df = pd.read_csv("results.csv")

print(df.sort_values("f1", ascending=False).head())
