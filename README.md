# 🏠 California House Price Prediction

A comprehensive machine learning project that predicts median house prices in California using various demographic and geographic features. This project demonstrates the complete end-to-end machine learning workflow, from data exploration to model deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Highlights](#key-highlights)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a machine learning pipeline to predict median house values in California districts. The solution uses multiple algorithms and demonstrates best practices in data preprocessing, feature engineering, model selection, and hyperparameter tuning.

## ✨ Features

- **Data Exploration & Visualization**: Comprehensive EDA with correlation analysis and data distribution plots
- **Advanced Preprocessing**: Custom transformers and sklearn pipelines for data cleaning
- **Feature Engineering**: Creation of derived features (rooms per household, bedrooms per room, etc.)
- **Stratified Sampling**: Ensures representative train/test splits based on income categories
- **Multiple Models**: Comparison of Linear Regression, Decision Tree, and Random Forest models
- **Hyperparameter Tuning**: GridSearchCV for optimal model configuration
- **Model Evaluation**: Cross-validation and confidence intervals for robust performance assessment
- **Model Persistence**: Saved trained model for future predictions

## 📊 Dataset

The project uses the California Housing dataset, which contains 20,640 observations with the following features:

- **Geographic Features**: `longitude`, `latitude`
- **Demographic Features**: `population`, `households`
- **Housing Features**: `housing_median_age`, `total_rooms`, `total_bedrooms`
- **Economic Features**: `median_income`
- **Categorical Features**: `ocean_proximity`
- **Target Variable**: `median_house_value`

## 🔬 Methodology

### 1. Data Loading & Exploration
- Loaded dataset from publicly available source
- Performed exploratory data analysis (EDA)
- Analyzed feature distributions and correlations
- Identified missing values and data quality issues

### 2. Data Preprocessing
- **Missing Value Imputation**: Filled missing values using median strategy
- **Categorical Encoding**: Applied One-Hot Encoding to `ocean_proximity`
- **Feature Engineering**: Created new features:
  - `rooms_per_household` = total_rooms / households
  - `bedrooms_per_room` = total_bedrooms / total_rooms
  - `population_per_household` = population / households
- **Feature Scaling**: Standardized numerical features using StandardScaler

### 3. Data Splitting
- Implemented stratified sampling based on income categories
- Split data into training (80%) and test (20%) sets
- Ensured representative distribution across income strata

### 4. Model Training & Selection
- **Linear Regression**: Baseline model for comparison
- **Decision Tree Regressor**: Captured non-linear patterns
- **Random Forest Regressor**: Ensemble method for improved performance
- Used 10-fold cross-validation for robust evaluation

### 5. Hyperparameter Tuning
- Applied GridSearchCV to find optimal hyperparameters
- Tuned `n_estimators` and `max_features` for Random Forest
- Evaluated multiple combinations with 5-fold cross-validation

### 6. Model Evaluation
- Final evaluation on held-out test set
- Computed RMSE and confidence intervals
- Assessed model generalization performance

## 🛠 Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms and preprocessing
- **SciPy**: Statistical analysis
- **joblib**: Model serialization

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Ade-omole/House_price_prediction.git
cd House_price_prediction
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook House_pricing.ipynb
```

2. Run all cells to:
   - Load and explore the data
   - Preprocess and engineer features
   - Train and evaluate models
   - Save the final model

### Using the Trained Model

The trained model (`my_final_model.pkl`) can be loaded and used for predictions:

```python
import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
model = joblib.load("my_final_model.pkl")

# Prepare your data (must match the training data format)
# Use the same preprocessing pipeline as in the notebook
predictions = model.predict(prepared_data)
```

## 📈 Results

### Model Performance Comparison

| Model | Cross-Validation RMSE | Standard Deviation |
|-------|----------------------|-------------------|
| Linear Regression | ~68,144 | ~1,057 |
| Decision Tree | ~68,851 | ~2,524 |
| **Random Forest** | **~48,347** | **~902** |

### Final Model Performance

- **Test Set RMSE**: ~48,589
- **95% Confidence Interval**: [46,290 - 50,783]

The Random Forest model achieved the best performance with significantly lower RMSE and more stable predictions (lower standard deviation).

## 📁 Project Structure

```
House_price_prediction/
│
├── House_pricing.ipynb      # Main Jupyter notebook with complete workflow
├── my_final_model.pkl       # Saved trained Random Forest model
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## 🌟 Key Highlights

- **End-to-End ML Pipeline**: Complete workflow from raw data to trained model
- **Best Practices**: Proper train/test split, cross-validation, and hyperparameter tuning
- **Custom Transformers**: Implementation of reusable data transformation classes
- **Feature Engineering**: Domain knowledge applied to create meaningful features
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Reproducible Research**: Well-documented code with clear explanations

## 🔮 Future Improvements

- [ ] Implement additional models (XGBoost, LightGBM, Neural Networks)
- [ ] Add more sophisticated feature engineering techniques
- [ ] Create a web application for real-time predictions
- [ ] Implement model versioning and MLOps practices
- [ ] Add automated testing for the ML pipeline
- [ ] Deploy model as a REST API
- [ ] Implement model monitoring and drift detection

## 📚 Acknowledgements

This project is inspired by concepts from **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron (3rd Edition).

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Adekanmi Omole

---

⭐ If you find this project helpful, please consider giving it a star!
