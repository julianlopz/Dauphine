import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min



stations = pl.read_csv("data/output/stations_df.csv")



#Chose variables to use
feature_columns = [
    "mean_duration",
    "median_duration",
    "std_duration",
    "n_destinations",
    "n_origins",
    "weekend_ratio",
    "morning_rush_ratio",
    "evening_rush_ratio",
    "night_ratio",
] + hour_columns

X = stations.select(feature_columns).to_numpy()

#standarize
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

###################
#PCA
##################

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

#Explained variance
explained_variance = pca.explained_variance_ratio_

for i, variance in enumerate(explained_variance[:10], start=1):
    print(f"PC{i}: {variance:.2%}")

#plot cumulative explained variance

plt.figure(figsize=(8, 5))

plt.plot(
    np.cumsum(explained_variance),
    marker="o"
)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA – cumulative explained variance")
plt.grid(alpha=0.3)

plt.show()



#Visualize stations in PCA space
plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    alpha=0.7,
    s=30
)

plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
plt.title("Station profiles – PCA")
plt.grid(alpha=0.2)

plt.show()


####################33
#To interpret pca components
####################3

#Get loadings
loadings = pl.DataFrame({
    "feature": feature_columns,
    "PC1": pca.components_[0],
    "PC2": pca.components_[1],
    "PC3": pca.components_[2],
})

#sort PC1
print(
    loadings
    .sort("PC1", descending=True)
)

#And separately look at the largest absolute contributions:
pc1 = (
    loadings
    .with_columns(
        pl.col("PC1").abs().alias("abs_PC1")
    )
    .sort("abs_PC1", descending=True)
)

print(pc1)

#and pc2
pc2 = (
    loadings
    .with_columns(
        pl.col("PC2").abs().alias("abs_PC2")
    )
    .sort("abs_PC2", descending=True)
)

print(pc2)


#########################33
#Non linear dimension reduction: t-SNE
###########################

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_scaled)

#plot
plt.figure(figsize=(10, 7))

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    s=30,
    alpha=0.7
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Station profiles – t-SNE")
plt.grid(alpha=0.2)

plt.show()


#And using umap
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    s=30,
    alpha=0.7
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Station profiles – UMAP")
plt.grid(alpha=0.2)

plt.show()

####################3
#Identify natural groups
######################
from sklearn.metrics import silhouette_score

results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append((k, score))

print(results)



ks = [x[0] for x in results]
scores = [x[1] for x in results]

plt.figure(figsize=(8, 5))

plt.plot(ks, scores, marker="o")

plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Choice of number of clusters")

plt.show()


#test several K values, for example
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X_scaled)

################3
#Color PCA plot by cluster
#################

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="tab10",
    s=40,
    alpha=0.8
)

plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
plt.title("Station profiles – PCA + K-means")

plt.colorbar(scatter, label="Cluster")

plt.show()


#and same for umap
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=labels,
    cmap="tab10",
    s=40,
    alpha=0.8
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Station profiles – UMAP + K-means")

plt.colorbar(scatter, label="Cluster")

plt.show()


##############
#Find atypical or isolated stations
###############

closest, distances = pairwise_distances_argmin_min(
    X_scaled,
    kmeans.cluster_centers_
)

stations_analysis = stations.with_columns([
    pl.Series("cluster", labels),
    pl.Series("distance_to_cluster_center", distances)
])

print(
    stations_analysis
    .sort("distance_to_cluster_center", descending=True)
    .select([
        "station",
        "cluster",
        "distance_to_cluster_center"
    ])
    .head(10)
)


#################33
#Make clusters interpretables
##################

cluster_summary = (
    stations_analysis
    .group_by("cluster")
    .agg([
        pl.col(c).mean().alias(c)
        for c in feature_columns
    ])
)

print(cluster_summary)

