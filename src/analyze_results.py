from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------

# 1. Locate project files

# --------------------------------------------------

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "outputs" / "results"
CHARTS_DIR = PROJECT_DIR / "outputs" / "charts"

DATA_PATH = RESULTS_DIR / "customer_monthly_spending.csv"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("WEEK 5: MODEL DIAGNOSTICS")
print("=" * 65)

# --------------------------------------------------

# 2. Load data

# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
pred = pd.read_csv(PREDICTIONS_PATH)

df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
pred["year_month"] = pd.PeriodIndex(pred["year_month"], freq="M")

print("\nModeling dataset rows:", f"{len(df):,}")
print("Test prediction rows:", f"{len(pred):,}")

# --------------------------------------------------

# 3. Inspect monthly target distribution

# --------------------------------------------------

print("\nTARGET DISTRIBUTION")
print("-" * 65)

target = df["target_next_month_spending"]

print(target.describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]))

# --------------------------------------------------

# 4. Inspect test prediction errors

# --------------------------------------------------

actual = pred["target_next_month_spending"].to_numpy()
rf_pred = pred["random_forest_prediction"].to_numpy()

errors = actual - rf_pred

diagnostics = pd.DataFrame(
{
"actual": actual,
"prediction": rf_pred,
"absolute_error": np.abs(errors),
"squared_error": errors ** 2,
}
)

print("\nRANDOM FOREST TEST DIAGNOSTICS")
print("-" * 65)

print("MAE:", round(mean_absolute_error(actual, rf_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(actual, rf_pred)), 2))
print("R2:", round(r2_score(actual, rf_pred), 4))

print("\nActual spending percentiles:")
print(
pd.Series(actual)
.describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99])
)

print("\nLargest absolute prediction errors:")
print(diagnostics.nlargest(10, "absolute_error").to_string(index=False))

# --------------------------------------------------

# 5. Performance by target-spending group

# --------------------------------------------------

diagnostics["spending_group"] = pd.cut(
diagnostics["actual"],
bins=[-np.inf, 250, 500, 1000, 5000, np.inf],
labels=[
"Up to 250",
"250-500",
"500-1,000",
"1,000-5,000",
"Above 5,000",
],
)

group_results = (
diagnostics.groupby("spending_group", observed=False)
.agg(
records=("actual", "size"),
mean_actual=("actual", "mean"),
mean_prediction=("prediction", "mean"),
MAE=("absolute_error", "mean"),
)
.reset_index()
)

print("\nERROR BY ACTUAL SPENDING GROUP")
print("-" * 65)
print(group_results.to_string(index=False))

group_results.to_csv(
RESULTS_DIR / "error_by_spending_group.csv",
index=False,
)

# --------------------------------------------------

# 6. Plot target distribution

# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(target, bins=60, edgecolor="black")

plt.xlabel("Next-Month Spending")
plt.ylabel("Frequency")
plt.title("Distribution of Next-Month Customer Spending")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "target_distribution.png",
dpi=300,
)

plt.close()

# --------------------------------------------------

# 7. Plot actual versus predicted values

# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(actual, rf_pred, alpha=0.4)

max_value = max(float(np.max(actual)), float(np.max(rf_pred)))

plt.plot(
[0, max_value],
[0, max_value],
linestyle="--",
)

plt.xlabel("Actual Spending")
plt.ylabel("Predicted Spending")
plt.title("Random Forest: Actual vs Predicted")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "actual_vs_predicted_diagnostic.png",
dpi=300,
)

plt.close()

# --------------------------------------------------

# 8. Save diagnostics

# --------------------------------------------------

diagnostics.to_csv(
RESULTS_DIR / "prediction_diagnostics.csv",
index=False,
)

print("\nDiagnostic outputs saved:")
print("- outputs/results/error_by_spending_group.csv")
print("- outputs/results/prediction_diagnostics.csv")
print("- outputs/charts/target_distribution.png")
print("- outputs/charts/actual_vs_predicted_diagnostic.png")

print("\nModel diagnostics completed!")
