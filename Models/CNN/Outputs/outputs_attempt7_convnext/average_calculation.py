import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONFIG — change this to your actual predictions file path
# ---------------------------------------------------------
input_csv  = "/home/ahmed/project/outputs_attempt7_convnext/test_predictions.csv"
output_csv = "/home/ahmed/project/outputs_attempt7_convnext/test_predictions_with_pct_error.csv"
summary_txt = "/home/ahmed/project/outputs_attempt7_convnext/pct_error_summary.txt"

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
df = pd.read_csv(input_csv)

# Clip to avoid division by zero
actual = df["actual_views"].clip(lower=1)

# ---------------------------------------------------------
# COMPUTE PERCENTAGE ERRORS
# ---------------------------------------------------------
df["pct_error"] = (df["predicted_views"] - df["actual_views"]) / actual * 100
df["abs_pct_error"] = df["pct_error"].abs()

# ---------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------
mean_abs = df["abs_pct_error"].mean()
median_abs = df["abs_pct_error"].median()
p25 = df["abs_pct_error"].quantile(0.25)
p75 = df["abs_pct_error"].quantile(0.75)
p90 = df["abs_pct_error"].quantile(0.90)
p95 = df["abs_pct_error"].quantile(0.95)

summary = f"""
Percentage Error Summary
========================
Mean absolute % error:   {mean_abs:.2f}%
Median absolute % error: {median_abs:.2f}%

Percentiles:
  25th percentile: {p25:.2f}%
  75th percentile: {p75:.2f}%
  90th percentile: {p90:.2f}%
  95th percentile: {p95:.2f}%

Total samples: {len(df)}
"""

print(summary)

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
df.to_csv(output_csv, index=False)

with open(summary_txt, "w") as f:
    f.write(summary)

print(f"\nSaved full predictions with errors → {output_csv}")
print(f"Saved summary text file → {summary_txt}")
