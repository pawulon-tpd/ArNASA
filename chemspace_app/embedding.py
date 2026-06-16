"""Dimensionality reduction and clustering of the fingerprint space."""
import hdbscan
import umap


def compute_umap(fingerprints, n_neighbors=15, min_dist=0.1, random_state=42):
    n_samples = fingerprints.shape[0]
    n_neighbors = max(2, min(n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="jaccard",
        random_state=random_state,
    )
    return reducer.fit_transform(fingerprints)


def compute_clusters(fingerprints, min_cluster_size=5):
    n_samples = fingerprints.shape[0]
    min_cluster_size = max(2, min(min_cluster_size, n_samples))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="jaccard")
    return clusterer.fit_predict(fingerprints.astype(bool))
