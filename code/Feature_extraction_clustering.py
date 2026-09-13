import os
import cv2
import numpy as np
import h5py
from multiprocessing import Pool, cpu_count

# feature extraction process from dataset AID for the determination of centroids positions through clustering,
# aim to find a set of visual word to describe images


def feature_extraction(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error image name: {img_path}")
        return None

    sift_local = cv2.SIFT_create(nfeatures=200)  # 200 descriptors for image 600x600
    _, descriptors = sift_local.detectAndCompute(img, None)
    if descriptors is None:
        print(f"Error no descriptor: {img_path}")
        return None
    # regularitation L2
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors /= norm
    return descriptors


if __name__ == "__main__":
    with h5py.File("files/sift_descriptors.h5", "w") as hf:

        # inizialization of the storage dataset
        dataset = hf.create_dataset(
            "descriptors",
            shape=(0, 128),
            maxshape=(None, 128),
            chunks=True,
            dtype="float32",
        )

        n_cores = cpu_count() - 2
        base_path = "../AID"  # train dataset
        image_directories = sorted(os.listdir(base_path))
        # multiprocess computation
        with Pool(processes=n_cores) as pool:
            for folder in image_directories:
                folder_path = os.path.join(base_path, folder)
                if not os.path.isdir(folder_path):
                    print(f"Error in folder: {folder}")
                    continue
                print(f"Processing folder: {folder}")
                image_names = []
                for img in os.listdir(folder_path):
                    if img.lower().endswith(".jpg"):
                        image_path = os.path.join(folder_path, img)
                        image_names.append(image_path)
                results = pool.map(feature_extraction, image_names)
                folder_descriptors_list = [res for res in results if res is not None]

                if folder_descriptors_list:
                    # matrix (n_descriptors x 128)
                    folder_data = np.vstack(folder_descriptors_list)
                    # resizing of the dataset
                    current_rows = dataset.shape[0]
                    new_rows = folder_data.shape[0]
                    dataset.resize(current_rows + new_rows, axis=0)
                    # data saving
                    dataset[current_rows:] = folder_data
