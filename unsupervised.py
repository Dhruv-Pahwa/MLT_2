import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample dataset: Word frequency and article length
data = {
    'goal':    [5, 4, 0, 0, 0, 1],
    'vote':    [0, 1, 6, 5, 0, 0],
    'startup': [0, 0, 0, 0, 7, 6],
    'length':  [500, 540, 420, 390, 600, 580]
}
df = pd.DataFrame(data)

# Clustering: Find 3 groups using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.drop('cluster', axis=1))
df['Topic Dimension 1'] = pca_result[:, 0]
df['Topic Dimension 2'] = pca_result[:, 1]

# Plotting the clusters
plt.figure(figsize=(7, 5))
for cluster in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster]
    plt.scatter(subset['Topic Dimension 1'], subset['Topic Dimension 2'], label=f'Cluster {cluster}', s=100)

plt.xlabel("Topic Dimension 1")
plt.ylabel("Topic Dimension 2")
plt.title("Unsupervised Learning: Clustering News Articles")
plt.legend()
plt.grid(True)
plt.show()
