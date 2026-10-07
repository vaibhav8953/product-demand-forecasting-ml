# Product Demand Classification Using Machine Learning

A Python project exploring retail inventory data and comparing machine learning models for classifying high and low product demand. Developed by **Vaibhav Mishra** as MSc coursework at Loughborough University London.

## Project objective

Explore sales and inventory patterns and predict whether a record belongs to the high-demand category:

- **High demand (1):** Units Sold > 150.
- **Low demand (0):** Units Sold ≤ 150.

Despite the repository's forecasting name, the current notebook performs demand classification rather than forecasting future sales quantities.

## Dataset

`inventory.csv` contains 73,100 rows and 15 columns, including dates, store and product identifiers, categories, regions, inventory levels, units sold and ordered, prices, discounts, weather, promotions, competitor pricing and seasonality.

The target is derived from Units Sold. Units Sold, Date, Demand Forecast and the target column are excluded from the model inputs.

## Workflow

1. Inspect the dataset, missing values and duplicate rows.
2. Convert dates and encode categorical variables.
3. Explore sales trends, distributions, correlations and inventory patterns.
4. Create the binary demand target.
5. Split the data into 80% training and 20% testing, using random_state=42.
6. Scale features for Logistic Regression.
7. Train and compare classification models.
8. Evaluate predictions and inspect Random Forest feature importance.

## Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- Stacking ensemble combining Random Forest and Gradient Boosting with Logistic Regression as the final estimator

## Evaluation

The notebook compares accuracy and ROC curves across models. It also displays a Logistic Regression confusion matrix and classification report, and Random Forest feature importance.

## Repository files

| File | Description |
| --- | --- |
| `F215020_Product_forecast_using_ML_3.5.ipynb` | Data exploration, preprocessing, modelling and evaluation |
| `inventory.csv` | Input dataset |
| `README.md` | Project overview and setup instructions |

## How to run

1. Download or clone the repository.
2. Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

3. Open a terminal in the repository folder and start Jupyter:

```bash
jupyter notebook
```

4. Open `F215020_Product_forecast_using_ML_3.5.ipynb`.
5. Keep `inventory.csv` in the same folder as the notebook and use this dataset-loading line:

```python
df = pd.read_csv("inventory.csv")
```

6. Run the cells in order.

If a recent pandas version raises an error when calculating correlations with the Date column present, use `df.corr(numeric_only=True)` in the correlation cells.

## Limitations

- The 150-unit threshold is a fixed project definition of high demand.
- Evaluation uses a random train/test split, not a chronological forecasting test.
- Identifier and categorical encoding is performed before the split; a stronger validation design would fit preprocessing within training pipelines.
- Feature importance indicates predictive associations rather than causal effects.
- The dataset source and redistribution licence should be documented before public distribution.
- This is an educational analysis, not a deployed inventory planning system.

## Author

**Vaibhav Mishra**  
MSc Digital Finance & AI, 
