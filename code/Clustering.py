import h5py
import faiss
import numpy as np
import time

# building process of the dictionary of visual words


K = 1000
output_file = "files/BoW_K1000.h5"

if __name__ == "__main__":
    with h5py.File("files/sift_descriptors.h5", "r") as hf:
        # file loading on RAM
        dataset = np.ascontiguousarray(hf["descriptors"][:], dtype="float32")
    print(f"Descriptor Dataset: {dataset.shape}")
    n_descriptor, dimension = dataset.shape

    start_time = time.time()
    print(f"Start K-means Clustering, K={K}")
    kmeans = faiss.Kmeans(
        dimension,
        K,
        niter=100,
        nredo=8,
        verbose=True,
        max_points_per_centroid=n_descriptor,
    )
    kmeans.train(dataset)
    end_time = time.time()
    print(f"Clustering completed: {end_time - start_time:.2f} s")
    centroids = kmeans.centroids
    with h5py.File(output_file, "w") as hf_out:
        hf_out.create_dataset("centroids", data=centroids)
