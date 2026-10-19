from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================

# 1. PROJECT PATHS

# ============================================================

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_PATH = (
PROJECT_DIR
/ "outputs"
/ "results"
/ "customer_monthly_spending.csv"
)

RESULTS_DIR = PROJECT_DIR / "outputs" / "results"
CHARTS_DIR = PROJECT_DIR / "outputs" / "charts"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WEEK 5: CORRECTED MACHINE LEARNING EVALUATION")
print("=" * 70)

# ============================================================

# 2. LOAD DATA

# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")

# The row's features describe one month, while its target

# represents the following month's spending.

df["target_month"] = df["year_month"] + 1

df = df.sort_values(
["target_month", "customer_id"]
).reset_index(drop=True)

print("\nDataset loaded successfully!")
print("Total records:", f"{len(df):,}")
print("Distinct target months:", df["target_month"].nunique())

# ============================================================

# 3. DEFINE FEATURES AND TARGET

# ============================================================

feature_columns = [
"monthly_spending",
"monthly_orders",
"monthly_items",
"monthly_unique_products",
"active_days",
"previous_month_spending",
"spending_3month_average",
]

# The calendar month being predicted is known in advance.

df["target_year"] = df["target_month"].map(lambda p: p.year)
df["target_month_number"] = df["target_month"].map(lambda p: p.month)

feature_columns += ["target_year", "target_month_number"]

target_column = "target_next_month_spending"

# ============================================================

# 4. CHRONOLOGICAL TRAIN/TEST SPLIT BY TARGET MONTH

# ============================================================

all_target_months = np.array(
sorted(df["target_month"].unique())
)

if len(all_target_months) < 6:
    raise ValueError(
"Not enough distinct target months for temporal evaluation."
)

split_index = int(len(all_target_months) * 0.80)
split_index = min(
max(split_index, 1),
len(all_target_months) - 1,
)

cutoff_month = all_target_months[split_index]

train_mask = df["target_month"] < cutoff_month
test_mask = df["target_month"] >= cutoff_month

train_df = df.loc[train_mask].copy().reset_index(drop=True)
test_df = df.loc[test_mask].copy().reset_index(drop=True)

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df[target_column]
y_test = test_df[target_column]

print("\nCHRONOLOGICAL SPLIT BY TARGET MONTH")
print("-" * 70)
print("First target month:", all_target_months[0])
print("Test cutoff month:", cutoff_month)
print("Training target months:", train_df["target_month"].nunique())
print("Testing target months:", test_df["target_month"].nunique())
print("Training records:", f"{len(train_df):,}")
print("Testing records:", f"{len(test_df):,}")

if X_train.empty or X_test.empty:
    raise ValueError("Training or testing data is empty.")

# ============================================================

# 5. BUILD MONTH-BASED CROSS-VALIDATION FOLDS

# ============================================================

training_months = np.array(
sorted(train_df["target_month"].unique())
)

if len(training_months) < 4:
    raise ValueError(
"Not enough training months for time-series cross-validation."
)
time_splitter = TimeSeriesSplit(n_splits=3)

cv_folds = []

for earlier_month_indices, validation_month_indices in (
    time_splitter.split(training_months)
):
    earlier_months = training_months[earlier_month_indices]
    validation_months = training_months[validation_month_indices]

    fold_train_indices = np.flatnonzero(
        train_df["target_month"].isin(earlier_months).to_numpy()
    )

    fold_validation_indices = np.flatnonzero(
        train_df["target_month"].isin(validation_months).to_numpy()
    )

    cv_folds.append(
        (fold_train_indices, fold_validation_indices)
    )

print("\nMONTH-BASED CROSS-VALIDATION")
print("-" * 70)
print("Number of folds:", len(cv_folds))

print("\nMONTH-BASED CROSS-VALIDATION")
print("-" * 70)
print("Number of folds:", len(cv_folds))

for fold_number, (train_indices, validation_indices) in enumerate(
cv_folds, start=1
):
    fold_train_months = train_df.iloc[
train_indices
]["target_month"]


fold_validation_months = train_df.iloc[
    validation_indices
]["target_month"]

print(
    f"Fold {fold_number}: "
    f"training through {fold_train_months.max()}, "
    f"validating {fold_validation_months.min()} "
    f"to {fold_validation_months.max()}"
)

# ============================================================

# 6. LINEAR REGRESSION BASELINE

# ============================================================

print("\nTraining Linear Regression baseline...")

linear_model = TransformedTargetRegressor(
regressor=Pipeline(
steps=[
("scaler", StandardScaler()),
("model", LinearRegression()),
]
),
func=np.log1p,
inverse_func=np.expm1,
)

linear_model.fit(X_train, y_train)

linear_predictions = np.maximum(
linear_model.predict(X_test),
0,
)

# ============================================================

# 7. RANDOM FOREST HYPERPARAMETER TUNING

# ============================================================

print("\nPreparing Random Forest search...")

rf_model = TransformedTargetRegressor(
regressor=Pipeline(
steps=[
(
"model",
RandomForestRegressor(
random_state=42,
n_jobs=-1,
),
)
]
),
func=np.log1p,
inverse_func=np.expm1,
)

parameter_options = {
"regressor__model__n_estimators": [100, 150, 200],
"regressor__model__max_depth": [8, 12, 20, None],
"regressor__model__min_samples_split": [2, 5, 10],
"regressor__model__min_samples_leaf": [1, 2, 4],
"regressor__model__max_features": [0.7, 1.0],
}

search = RandomizedSearchCV(
estimator=rf_model,
param_distributions=parameter_options,
n_iter=6,
scoring="neg_mean_absolute_error",
cv=cv_folds,
random_state=42,
n_jobs=1,
verbose=1,
error_score="raise",
)

print("Parameter combinations:", search.n_iter)

search.fit(X_train, y_train)

best_rf = search.best_estimator_

rf_predictions = np.maximum(
best_rf.predict(X_test),
0,
)

print("\nBEST RANDOM FOREST PARAMETERS")
print(search.best_params_)

print(
"\nBest cross-validation MAE:",
round(-search.best_score_, 2),
)

# ============================================================

# 8. MODEL EVALUATION

# ============================================================

def evaluate_model(model_name, actual, predicted):
    return {
"Model": model_name,
"MAE": mean_absolute_error(actual, predicted),
"RMSE": np.sqrt(
mean_squared_error(actual, predicted)
),
"R2": r2_score(actual, predicted),
}

results = pd.DataFrame(
[
evaluate_model(
"Linear Regression",
y_test,
linear_predictions,
),
evaluate_model(
"Random Forest",
y_test,
rf_predictions,
),
]
)

print("\nFINAL MODEL EVALUATION")
print("=" * 70)
print(results.round(4).to_string(index=False))

results.to_csv(
RESULTS_DIR / "model_evaluation.csv",
index=False,
)

# ============================================================

# 9. SAVE BEST PARAMETERS

# ============================================================

with open(
RESULTS_DIR / "best_parameters.txt",
"w",
encoding="utf-8",
) as output_file:
    output_file.write(str(search.best_params_) + "\n")
    output_file.write(
f"Best cross-validation MAE: {-search.best_score_:.4f}\n"
)


# ============================================================

# 10. SAVE TEST PREDICTIONS

# ============================================================

predictions_df = test_df[
["customer_id", "year_month", "target_month", target_column]
].copy()

predictions_df["linear_regression_prediction"] = linear_predictions
predictions_df["random_forest_prediction"] = rf_predictions

predictions_df.to_csv(
RESULTS_DIR / "test_predictions.csv",
index=False,
)

# ============================================================

# 11. FEATURE IMPORTANCE

# ============================================================

fitted_rf_pipeline = best_rf.regressor_

importances = fitted_rf_pipeline.named_steps[
"model"
].feature_importances_

importance_df = pd.DataFrame(
{
"Feature": feature_columns,
"Importance": importances,
}
).sort_values(
"Importance",
ascending=False,
)

importance_df.to_csv(
RESULTS_DIR / "feature_importance.csv",
index=False,
)

print("\nFEATURE IMPORTANCE")
print(importance_df.round(4).to_string(index=False))

# ============================================================

# 12. ACTUAL VS PREDICTED CHART

# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
y_test,
rf_predictions,
alpha=0.45,
)

max_value = max(
float(y_test.max()),
float(rf_predictions.max()),
)

plt.plot(
[0, max_value],
[0, max_value],
linestyle="--",
)

plt.xlabel("Actual Next-Month Spending")
plt.ylabel("Predicted Next-Month Spending")
plt.title("Random Forest: Actual vs Predicted Spending")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "actual_vs_predicted.png",
dpi=300,
)

plt.close()

# ============================================================

# 13. RESIDUAL DISTRIBUTION

# ============================================================

residuals = (
y_test.to_numpy()
- rf_predictions
)

plt.figure(figsize=(9, 6))

plt.hist(
residuals,
bins=50,
edgecolor="black",
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Random Forest Residual Distribution")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "residual_distribution.png",
dpi=300,
)

plt.close()

# ============================================================

# 14. MODEL COMPARISON CHART

# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
results["Model"],
results["RMSE"],
)

plt.ylabel("RMSE")
plt.title("Model Comparison: Lower RMSE Is Better")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "model_comparison.png",
dpi=300,
)

plt.close()

# ============================================================

# 15. FEATURE IMPORTANCE CHART

# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
importance_df["Feature"][::-1],
importance_df["Importance"][::-1],
)

plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
CHARTS_DIR / "feature_importance.png",
dpi=300,
)

plt.close()

print("\nSAVED OUTPUTS")
print("-" * 70)
print("outputs/results/model_evaluation.csv")
print("outputs/results/best_parameters.txt")
print("outputs/results/test_predictions.csv")
print("outputs/results/feature_importance.csv")
print("outputs/charts/actual_vs_predicted.png")
print("outputs/charts/residual_distribution.png")
print("outputs/charts/model_comparison.png")
print("outputs/charts/feature_importance.png")

print("\nCorrected model training and evaluation completed!")
