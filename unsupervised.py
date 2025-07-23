from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
df = pd.DataFrame(X, columns=["TechWords", "PoliticsWords"])

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="TechWords", y="PoliticsWords", hue="Cluster", palette="Set2")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="X")
plt.title("Unsupervised Learning – News Article Clustering")
plt.show()
