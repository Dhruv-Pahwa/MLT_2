from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

articles = [
    "Team wins the football championship",
    "New political reforms announced",
    "Technology advances in AI",
    "Player breaks scoring record",
    "Parliament debates new law",
    "Breakthrough in quantum computing"
]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(articles)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

tsne = TSNE(n_components=2, perplexity=2, random_state=42)
X_2d = tsne.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i, (x, y) in enumerate(X_2d):
    plt.scatter(x, y, color=colors[labels[i]], s=100)
    plt.text(x + 0.02, y + 0.02, f"{i+1}: {articles[i][:30]}...", fontsize=8)

plt.title("Unsupervised Clustering of News Articles")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
