# tips-prediction-analysis
Predicting restaurant tips with ML: Linear Regression predicts tip amounts, Logistic Regression classifies big tips (>75th percentile).  Uses Seaborn's "tips" dataset with feature engineering (tip percentage, labeling), visualizations, and metrics (RMSE, Accuracy).

# Restaurant Tips Prediction Analysis

Predicting restaurant tips using Linear Regression (for tip amounts) and Logistic Regression (for classifying big tips >75th percentile).

 Project Overview
This project analyzes the Seaborn "tips" dataset to:
- Predict tip amounts (regression)
- Classify whether a tip is "big" or not (classification)

 Technologies Used
- Python 3
- pandas, matplotlib, seaborn
- scikit-learn

 Code Features
- Feature engineering: tip percentage calculation
- Binary label creation (big tip threshold = 75th percentile)
- Linear Regression for tip prediction
- Logistic Regression for classification
- Performance metrics: RMSE & Accuracy
- Data visualizations

 How to Run
```bash
git clone https://github.com/Tomzy2203/tips-prediction-analysis.git
cd tips-prediction-analysis
python tips_analysis.py
