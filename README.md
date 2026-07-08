# BigMart Sales Prediction — A Regression Problem

Predicting `Item_Outlet_Sales` for products across BigMart outlets using exploratory
data analysis, feature engineering, and a comparison of multiple regression algorithms.

##  Problem Statement

The data scientists at BigMart have collected sales data for 1559 products across
10 stores in different cities. The goal is to build a predictive model that estimates
the sales of each product at a particular store — helping BigMart understand the
properties of products and stores that play a key role in increasing sales.

##  Dataset

The dataset (`Train.csv` / `Test.csv`) contains 8523 training records with 12 columns:

| Feature | Description |
|---|---|
| Item_Identifier | Unique product ID |
| Item_Weight | Weight of the product |
| Item_Fat_Content | Whether the product is low fat or regular |
| Item_Visibility | % of total display area allocated to the product |
| Item_Type | Category the product belongs to |
| Item_MRP | Maximum Retail Price |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year the outlet was established |
| Outlet_Size | Size of the store (Small/Medium/High) |
| Outlet_Location_Type | Type of city the outlet is in |
| Outlet_Type | Grocery store or supermarket type |
| **Item_Outlet_Sales** | **Target variable** — sales of the product in that store |

Source: [BigMart Sales Data on Kaggle](https://www.kaggle.com/datasets/mrmorj/bigmart-sales-data)


##  Approach

1. **EDA** — inspected shape, dtypes, missing values, summary statistics, and
   correlation heatmap; visualized the (right-skewed) target distribution.
2. **Missing Value Treatment** — imputed `Item_Weight` with the mean and
   `Outlet_Size` with the mode.
3. **Data Cleaning & Feature Engineering**
   - Dropped ID columns with no predictive value (`Item_Identifier`, `Outlet_Identifier`)
   - Standardized inconsistent `Item_Fat_Content` labels (`LF`, `low fat` → `Low Fat`, etc.)
   - Encoded `Outlet_Size` and `Outlet_Location_Type` as ordinal integers
   - Engineered `Outlet_Age` from `Outlet_Establishment_Year`
   - Squared `Item_MRP` and `Item_Visibility` to strengthen their linear relationship with sales
   - One-hot encoded remaining categorical columns
4. **Outlier Treatment** — tested percentile-based clipping on `Item_Visibility`, but
   it reduced model performance, so it was left out of the final pipeline.
5. **Modeling** — trained and benchmarked 8 regression algorithms on an 80/20 split:
   - Linear Regression, Ridge, Lasso, ElasticNet
   - Extra Tree Regressor, Gradient Boosting Regressor
   - K-Neighbors Regressor, XGBoost Regressor

##  Results

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting Regressor** | **~0.60** | **~1038.2** | **~738.2** |
| Linear Regression | ~0.56 | — | — |
| Ridge Regression | ~0.56 | — | — |

Gradient Boosting Regressor gave the best results, followed closely by Linear and
Ridge Regression.

##  Tech Stack

- Python 3
- pandas, numpy — data manipulation
- seaborn, matplotlib — visualization
- scikit-learn — preprocessing, modeling, evaluation
- xgboost — gradient boosted trees


