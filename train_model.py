import os
import gzip
import urllib.request
import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# MNIST DATASET DOWNLOAD
# ============================================================

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}


def download_file(filename):
    """
    Download MNIST file if it does not already exist.
    """

    url = BASE_URL + filename

    if not os.path.exists(filename):

        print("Downloading:", filename)

        urllib.request.urlretrieve(
            url,
            filename
        )

        print("Download completed.")

    else:

        print("Already exists:", filename)


# ============================================================
# DOWNLOAD ALL MNIST FILES
# ============================================================

print()
print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNIZER")
print("=" * 60)
print()

for filename in FILES.values():

    download_file(filename)


# ============================================================
# READ MNIST IMAGES
# ============================================================

def load_images(filename):

    with gzip.open(filename, "rb") as f:

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=16
        )

    images = data.reshape(
        -1,
        28 * 28
    )

    return images


# ============================================================
# READ MNIST LABELS
# ============================================================

def load_labels(filename):

    with gzip.open(filename, "rb") as f:

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8,
            offset=8
        )

    return data


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading MNIST dataset...")

X_train = load_images(
    FILES["train_images"]
)

y_train = load_labels(
    FILES["train_labels"]
)

X_test = load_images(
    FILES["test_images"]
)

y_test = load_labels(
    FILES["test_labels"]
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# NORMALIZE PIXELS
# ============================================================

X_train = X_train.astype(
    np.float32
) / 255.0

X_test = X_test.astype(
    np.float32
) / 255.0


# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================

print()
print("Training neural network...")
print("This may take a few minutes.")
print()


model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="relu",
    solver="adam",
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=20,
    random_state=42,
    verbose=True
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST MODEL
# ============================================================

print()
print("Testing model...")

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print(
    "Model training completed successfully!"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "digit_recognizer_model.pkl"
)


print()
print("Model saved successfully!")
print(
    "File: digit_recognizer_model.pkl"
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)