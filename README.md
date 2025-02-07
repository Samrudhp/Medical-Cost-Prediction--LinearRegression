# Medical Insurance Cost Prediction Model

## Overview
This project implements a machine learning model to predict medical insurance costs based on various personal attributes. The model uses linear regression along with regularization techniques (Ridge and Lasso) to provide accurate cost predictions.

## Dataset Features
The model uses the following features to predict insurance charges:

- **age**: Age of the primary beneficiary
- **sex**: Gender of the insurance contractor (female/male)
- **bmi**: Body Mass Index (BMI)
- **children**: Number of children/dependents covered by the insurance
- **smoker**: Smoking status (yes/no)
- **region**: Residential area of the beneficiary (northeast, southeast, southwest, northwest)

## Model Performance
The model demonstrates strong predictive capabilities with the following metrics:

- **Linear Regression**:
  - MSE: 33,596,915.85
  - R-squared: 0.784 (78.4% variance explained)

- **Ridge Regression** (alpha = 1.0):
  - MSE: 33,620,268.92
  - R-squared: 0.783

- **Lasso Regression** (alpha = 0.1):
  - MSE: 33,597,196.12
  - R-squared: 0.784

## Technical Implementation

### Data Preprocessing
- Feature scaling using StandardScaler for numerical features
- One-hot encoding for categorical variables
- Implemented using scikit-learn's ColumnTransformer and Pipeline

### Model Pipeline
```python
Pipeline(steps=[
    ('preprocessor', ColumnTransformer),
    ('regressor', LinearRegression())
])
```

### Dependencies
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

## Usage
To use this model:

1. Ensure all dependencies are installed:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

2. Load your data:
```python
data = pd.read_csv('insurance.csv')
```

3. Train the model:
```python
model.fit(X_train, y_train)
```

4. Make predictions:
```python
predictions = model.predict(X_test)
```

## Model Evaluation
- Includes visualization of actual vs predicted values
- Residual analysis to verify model assumptions
- Comparison of different regularization techniques

## Future Improvements
- Feature engineering to capture more complex relationships
- Hyperparameter tuning using cross-validation
- Exploration of non-linear models for potentially better performance
