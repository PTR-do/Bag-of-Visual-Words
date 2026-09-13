import cv2
import joblib
import numpy as np
from build_histogram_function import *

# classifier for image on the 21 classes of the dataset UCMerced_LandUse


class image_classifier:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.final_svm = joblib.load("files/classification_model.joblib")
        self.index = build_index(500, "files/BoW_K500.h5")
        with h5py.File(f"files/histograms_K500.h5", "r") as hf:
            self.class_names = [n for n in hf.attrs["class_names"]]

    def feature_extraction(self, img):
        if img is None:
            print(f"Error, the file is not an image.")
            return None
        sift = cv2.SIFT_create(nfeatures=400)
        _, descriptors = sift.detectAndCompute(img, None)
        if descriptors is None:
            print(f"Error, no descriptor in the image")
            return None
        norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors /= norm
        return descriptors

    def predict(self):
        descriptors = self.feature_extraction(self.img)
        if descriptors is None:
            return "Try with a new image."
        histogram = build_histogram(descriptors, self.index, 500)
        y_pred = self.final_svm.predict(histogram.reshape(1, -1))
        return f"Class: {self.class_names[y_pred[0]]}"


if __name__ == "__main__":
    img = "../images/baseballdiamond.webp"
    classifier = image_classifier(img)
    print(classifier.predict())
