import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = sns.load_dataset("tips")
df['tip_percentage'] = df['tip'] / df['total_bill'] * 100


big_tip_threshold = df['tip'].quantile(0.75)
df['big_tip'] = (df['tip'] > big_tip_threshold).astype(int)

X = df[['total_bill', 'size', 'day', 'time']]
X = pd.get_dummies(X, drop_first=True)

y_reg = df['tip']
y_cls = df['big_tip']

X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, random_state=42)


reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

reg_rmse = mean_squared_error(y_reg_test, y_reg_pred, squared=False)

print("REGRESSION: Predicting Tip Amount")
print("="*40)
print(f"RMSE: ${reg_rmse:.2f} (Average tip: ${y_reg.mean():.2f})")

# Simple regression plot
plt.scatter(y_reg_test, y_reg_pred, alpha=0.7)
plt.plot([0, df['tip'].max()], [0, df['tip'].max()], 'r--')
plt.xlabel("Actual Tip ($)")
plt.ylabel("Predicted Tip ($)")
plt.title("Regression: Actual vs Predicted Tips")
plt.show()

cls_model = LogisticRegression(max_iter=200)
cls_model.fit(X_train, y_cls_train)
y_cls_pred = cls_model.predict(X_test)

accuracy = accuracy_score(y_cls_test, y_cls_pred)
print("\nCLASSIFICATION: Predicting Big Tip (>${:.2f})".format(big_tip_threshold))
print("="*40)
print(f"Accuracy: {accuracy*100:.1f}%")




