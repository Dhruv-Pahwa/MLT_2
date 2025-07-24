from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, n_classes=2, random_state=42)

df = pd.DataFrame(X, columns=["BloodSugar", "BMI"])
df["Diagnosis"] = y
df["Diagnosis"] = df["Diagnosis"].map({0: "Healthy", 1: "Disease Prone"})

model = LogisticRegression()
model.fit(X, y)

plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="BloodSugar", y="BMI", hue="Diagnosis", palette="Set1", s=70)

x_vals = X[:, 0]
y_vals = -(model.coef_[0][0] * x_vals + model.intercept_) / model.coef_[0][1]
plt.plot(x_vals, y_vals, color='black', linestyle='--', label="Decision Boundary")

plt.title("Supervised Learning – Disease Prediction")
plt.xlabel("Blood Sugar")
plt.ylabel("BMI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
