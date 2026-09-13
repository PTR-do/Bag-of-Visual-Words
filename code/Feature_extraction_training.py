import os
import cv2
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count
from build_histogram_function import *

# feature extraction and histograms building process from dataset UCMerced_LandUse for training


K = 500
centroids_file = f"files/BoW_k{K}.h5"
output_file = f"files/histograms_K{K}.h5"


# index structrure for every single worker of multiprocessing system
def init_worker(k, centroids_file):
    global index
    index = build_index(k, centroids_file)


def feature_extraction(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error image name: {img_path}")
        return None
    sift_local = cv2.SIFT_create(nfeatures=400)  # 400 descriptors for image 256x256
    _, descriptors = sift_local.detectAndCompute(img, None)
    if descriptors is None:
        print(f"Error no descriptor: {img_path}")
        return None
    return build_histogram(descriptors, index, K)


if __name__ == "__main__":

    n_cores = cpu_count() - 2
    base_path = "../UCMerced_LandUse/Images"  # train dataset
    image_directories = sorted(os.listdir(base_path))
    class_idx = {name: i for i, name in enumerate(image_directories)}

    with h5py.File(output_file, "w") as hf:
        # inizialization of the storage dataset
        histograms_dataset = hf.create_dataset(
            "histograms", shape=(0, K), maxshape=(None, K), chunks=True, dtype="float32"
        )
        # inizialization of the label storage
        class_name_dataset = hf.create_dataset(
            "class_name", shape=(0,), maxshape=(None,), chunks=True, dtype="int32"
        )
        hf.attrs["class_names"] = [name for name in image_directories]

        with Pool(
            processes=n_cores, initializer=init_worker, initargs=(K, centroids_file)
        ) as pool:
            for folder in image_directories:
                folder_path = os.path.join(base_path, folder)
                if not os.path.isdir(folder_path):
                    print(f"Error in folder: {folder}")
                    continue
                print(f"Processing folder: {folder}")
                image_names = []
                for img in os.listdir(folder_path):
                    if img.lower().endswith(".tif"):
                        image_path = os.path.join(folder_path, img)
                        image_names.append(image_path)
                results = pool.map(feature_extraction, image_names)
                histograms = [res for res in results if res is not None]

                if histograms:
                    # matrix (n_image x K)
                    folder_data = np.vstack(histograms)
                    # resizing of the dataset
                    current_rows = histograms_dataset.shape[0]
                    new_rows = folder_data.shape[0]
                    histograms_dataset.resize(current_rows + new_rows, axis=0)
                    # data saving
                    histograms_dataset[current_rows:] = folder_data

                    # label of the images saving
                    current_rows = class_name_dataset.shape[0]
                    new_rows = folder_data.shape[0]
                    class_name_dataset.resize(current_rows + new_rows, axis=0)
                    class_name_dataset[current_rows:] = class_idx[folder]
