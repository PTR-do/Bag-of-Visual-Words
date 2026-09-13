import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# training of two classifier model (Random Forest and SVM) and comparison of the result
# for dictionary of 50, 100, 500 and 1000 visual words


K = 500

if __name__ == "__main__":
    # dataset loading
    with h5py.File(f"files/histograms_K{K}.h5", "r") as hf:
        data = hf["histograms"][:]
        label = hf["class_name"][:]
        class_names = [n for n in hf.attrs["class_names"]]
        num_classes = len(class_names)
    print(f"Data shape: {data.shape}\nlabel shape: {label.shape}")

    # 3 fold validation
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=1)
    svm = SVC(kernel="rbf", C=10.0, gamma=1, random_state=1)
    accuracy_rf, accuracy_svm = [], []
    cm_rf = np.zeros((num_classes, num_classes), dtype=int)
    cm_svm = np.zeros((num_classes, num_classes), dtype=int)

    print("Start 3-fold validation")
    for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = label[train_index], label[test_index]

        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        accuracy_rf.append(accuracy_score(y_test, y_pred_rf))
        cm_rf += confusion_matrix(y_test, y_pred_rf, labels=np.arange(num_classes))

        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        accuracy_svm.append(accuracy_score(y_test, y_pred_svm))
        cm_svm += confusion_matrix(y_test, y_pred_svm, labels=np.arange(num_classes))
        print(f"Fold {i+1} completato.")

    # results visualization
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].set_title(
        f"Confusion Matrix Random Forest\nAccuracy: {np.mean(accuracy_rf):.4f}"
    )
    sns.heatmap(
        cm_rf,
        annot=True,
        cbar=False,
        cmap="Greens",
        ax=ax[0],
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax[1].set_title(f"Confusion Matrix SVM\nAccuracy: {np.mean(accuracy_svm):.4f}")
    sns.heatmap(
        cm_svm,
        annot=True,
        cbar=False,
        cmap="Blues",
        ax=ax[1],
        xticklabels=class_names,
        yticklabels=class_names,
    )
    fig.tight_layout()
    fig.savefig(f"results_k{K}.png")
    plt.show()

    # final model
    final_svm = SVC(kernel="rbf", C=10.0, gamma=1)
    final_svm.fit(data, label)
    joblib.dump(final_svm, "files/classification_model.joblib")
    fsvm_pred = final_svm.predict(data)
    cm_fsvm = confusion_matrix(label, fsvm_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_fsvm,
        cbar=False,
        annot=True,
        cmap="coolwarm",
        xticklabels=class_names,
        yticklabels=class_names,
        fmt="0",
    )
    plt.show()
