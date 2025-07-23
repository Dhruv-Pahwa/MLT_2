from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=["BloodSugar", "BMI"])
df["Diagnosis"] = y

model = LogisticRegression()
model.fit(X, y)

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="BloodSugar", y="BMI", hue="Diagnosis", palette="Set1")
y_vals = -(model.coef_[0][0] * X[:, 0] + model.intercept_) / model.coef_[0][1]
plt.plot(X[:, 0], y_vals, color='black', linestyle='--')
plt.title("Supervised Learning – Disease Prediction")
plt.show()
