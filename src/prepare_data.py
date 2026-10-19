from pathlib import Path
import os

import pandas as pd

# --------------------------------------------------

# 1. Project paths

# --------------------------------------------------

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "online_retail_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "customer_monthly_spending.csv"

print("=" * 65)
print("WEEK 5: CUSTOMER SPENDING FEATURE ENGINEERING")
print("=" * 65)

# --------------------------------------------------

# 2. Load the cleaned retail dataset

# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("\nLoading retail dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Original rows:", f"{len(df):,}")

# --------------------------------------------------

# 3. Parse invoice dates

# --------------------------------------------------

df["invoicedate"] = pd.to_datetime(
df["invoicedate"],
errors="coerce"
)

df = df.dropna(subset=["invoicedate", "customer_id"])

# --------------------------------------------------

# 4. Keep positive, non-cancelled purchases

# --------------------------------------------------

df["invoice"] = df["invoice"].astype("string").str.strip()

df = df[
~df["invoice"].str.startswith("C", na=False)
& (df["quantity"] > 0)
& (df["price"] > 0)
& (df["transaction_value"] > 0)
].copy()

print("Valid purchase rows:", f"{len(df):,}")
print("Unique customers:", f"{df['customer_id'].nunique():,}")

# --------------------------------------------------

# 5. Aggregate transactions by customer and month

# --------------------------------------------------

df["customer_id"] = df["customer_id"].astype("int64")

df["year_month"] = df["invoicedate"].dt.to_period("M")

monthly = (
df.groupby(["customer_id", "year_month"], as_index=False)
.agg(
monthly_spending=("transaction_value", "sum"),
monthly_orders=("invoice", "nunique"),
monthly_items=("quantity", "sum"),
monthly_unique_products=("stockcode", "nunique"),
active_days=("invoicedate", lambda x: x.dt.date.nunique()),
)
)

monthly = monthly.sort_values(
["customer_id", "year_month"]
).reset_index(drop=True)

# --------------------------------------------------

# 6. Create next-month spending target

# --------------------------------------------------

monthly["target_next_month_spending"] = (
monthly.groupby("customer_id")["monthly_spending"]
.shift(-1)
)

monthly["next_month"] = (
monthly.groupby("customer_id")["year_month"]
.shift(-1)
)

# Keep only consecutive customer-month pairs.

# A target is valid only if it represents the immediately

# following calendar month.

monthly["expected_next_month"] = monthly["year_month"] + 1

monthly = monthly[
monthly["next_month"] == monthly["expected_next_month"]
].copy()

# --------------------------------------------------

# 7. Create historical features

# --------------------------------------------------

monthly["year"] = monthly["year_month"].map(lambda x: x.year)
monthly["month"] = monthly["year_month"].map(lambda x: x.month)

monthly["previous_month_spending"] = monthly["monthly_spending"]

monthly["spending_3month_average"] = (
monthly.groupby("customer_id")["monthly_spending"]
.transform(
lambda s: s.shift(1).rolling(3, min_periods=1).mean()
)
)

# The current month's observed spending is available at

# prediction time and is also a useful historical feature.

monthly["spending_3month_average"] = monthly[
"spending_3month_average"
].fillna(monthly["monthly_spending"])

# --------------------------------------------------

# 8. Select model features and target

# --------------------------------------------------

feature_columns = [
"monthly_spending",
"monthly_orders",
"monthly_items",
"monthly_unique_products",
"active_days",
"previous_month_spending",
"spending_3month_average",
"year",
"month",
]

model_data = monthly[
feature_columns + ["target_next_month_spending", "customer_id", "year_month"]
].copy()

model_data = model_data.dropna(
subset=["target_next_month_spending"]
)

model_data = model_data[
model_data["target_next_month_spending"] >= 0
].copy()

# --------------------------------------------------

# 9. Save the modeling dataset

# --------------------------------------------------

model_data["year_month"] = model_data["year_month"].astype(str)

model_data.to_csv(OUTPUT_PATH, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("-" * 65)
print("Modeling rows:", f"{len(model_data):,}")
print("Unique customers:", f"{model_data['customer_id'].nunique():,}")
print("Features:", feature_columns)
print("Target: target_next_month_spending")
print("Output file:", OUTPUT_PATH)

print("\nFirst five records:")
print(model_data.head().to_string(index=False))

print("\nTarget summary:")
print(model_data["target_next_month_spending"].describe())

print("\nSaved successfully!")
