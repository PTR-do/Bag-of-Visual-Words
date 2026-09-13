import faiss
import numpy as np
import h5py

# helper function for histogram building process from images


def build_index(K, centroids_file):
    with h5py.File(centroids_file, "r") as hf:
        centroids = hf["centroids"][:].astype("float32")
    index = faiss.IndexFlatL2(128)
    index.add(centroids)
    return index


def build_histogram(descriptors, index, K):
    if descriptors is None:
        return np.zeros(K)
    # regularitation L2 on descriptors
    descriptors = descriptors.astype("float32")
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors /= norm

    distances, position = index.search(descriptors, 1)
    hist, _ = np.histogram(position, bins=range(K + 1))
    hist = hist.astype("float32")
    if np.sum(hist) != 0:
        # regularitation L2 on histogram
        hist /= np.linalg.norm(hist)
    return hist
