# Machine Learning Algorithms and Model Evaluation

## Project Overview

This project was developed as part of my Week 5 Data Analytics internship with YUVA. The objective was to implement machine learning algorithms, prepare transactional data for predictive modeling, compare model performance, and evaluate the results using appropriate regression metrics.

Using the Online Retail dataset, this project develops a customer spending prediction model to estimate the next month's spending based on historical purchasing behavior.

## Objectives

* Prepare and transform transactional retail data for machine learning.
* Perform feature engineering to extract customer purchasing patterns.
* Implement and compare Linear Regression and Random Forest regression models.
* Apply chronological train-test splitting to evaluate performance on later periods.
* Tune Random Forest hyperparameters using time-based cross-validation.
* Evaluate predictions using MAE, RMSE, and R².
* Visualize model performance and identify important predictive features.

## Dataset

The project uses a cleaned Online Retail II dataset containing historical retail transactions from 2009–2011.

The original cleaned dataset contains approximately 1 million transaction records. For modeling, the data was filtered to retain valid purchase transactions and aggregated by customer and month.

The final modeling dataset contains **9,374 customer-month observations**.

The original dataset is not included in this repository because of its large file size.

## Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Git and GitHub

## Machine Learning Approach

Two regression algorithms were implemented:

1. **Linear Regression:** Used as a baseline model.
2. **Random Forest Regressor:** Tuned and evaluated to capture more complex relationships in customer spending.

Features include monthly spending, order counts, item quantities, unique products, active days, historical spending, and calendar information.

A chronological train-test split was used to respect the time order of the data and reduce the risk of evaluating the model on earlier periods than those used for training.

## Model Evaluation

The final evaluation produced the following results:

| Metric                         | Linear Regression | Random Forest |
| ------------------------------ | ----------------: | ------------: |
| Mean Absolute Error (MAE)      |           1681.03 |        589.86 |
| Root Mean Squared Error (RMSE) |          20437.53 |       2620.47 |
| R² Score                       |            -39.50 |         0.334 |

Random Forest performed substantially better than Linear Regression on the test set. Its R² score of approximately 0.334 indicates that the model explains some variation in next-month customer spending, but its predictive performance is still limited.

The model also struggles with unusually high-spending customers, whose spending patterns can produce large prediction errors. Therefore, the model should be treated as an experimental predictive model rather than a production-ready forecasting solution.

## Visualizations

The project includes visualizations for:

* Actual versus predicted customer spending
* Residual error distribution
* Model performance comparison
* Random Forest feature importance
* Target spending distribution

These visualizations help interpret model behavior, compare algorithms, and identify areas for improvement.

## Project Structure

```text
yuva_project5_machine_learning/
├── data/
```
