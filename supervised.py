import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Blood_Sugar_Level': [90, 150, 85, 200, 130, 170, 95, 145, 180, 110, 160, 175, 155],
    'BMI': [22, 35, 21, 40, 33, 38, 23, 34, 39, 25, 36, 37, 32],
    'Diabetes': [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
})

X = df[['Blood_Sugar_Level', 'BMI']]
y = df['Diabetes']

model = LogisticRegression()
model.fit(X, y)

colors = ['green' if d == 0 else 'red' for d in y]
plt.figure(figsize=(7, 5))
plt.scatter(X['Blood_Sugar_Level'], X['BMI'], c=colors, edgecolors='k')

x_min, x_max = X['Blood_Sugar_Level'].min() - 10, X['Blood_Sugar_Level'].max() + 10
y_min, y_max = X['BMI'].min() - 5, X['BMI'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.5))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn_r)

plt.xlabel("Blood Sugar Level")
plt.ylabel("BMI")
plt.title("Supervised Learning: Disease Prediction (Updated Sample)")
plt.grid(True)
plt.show()
