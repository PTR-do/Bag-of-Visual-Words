Bag of Visual Words Project.
You can find more information about Bag of Visual Words and the experiment conducted for the project in the Power Point presentation in the pdf.


The datasets used for this project are available at the links:
https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets
https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset


The code is structured as follows:

- Feature_extraction_clustering.py: feature extraction process from dataset AID for the determination 
  of centroids positions through clustering, aim to find a set of visual word to describe image.

- Clustering.py: building process of the dictionary of visual words.

- build_histogram_function.py: helper function for histogram building process to provide a standard description for images.

- Feature_extraction_training.py: feature extraction and histograms building process from dataset UCMerced_LandUse for training.

- Training.py: training of two shallow classifier (Random Forest and SVM) and comparison of the result.

- Image_classifier.py: final classifier for inference.


The file produced and saved during the experiment are the follows:

- sift_descriptors_h5: contain 2.000.000 descriptor from the first dataset to find visual word for the dictionary.

- BoW_K.h5: contains centroids positions for different dictionary size.

- histograms_K.h5: contains histograms of the training dataset for different dictionary size.


