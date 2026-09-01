"""
Question Set H - Convolutional Neural Network (CNN)
Dataset: MNIST Handwritten Digits
Keras/TensorFlow is used here (standard tool for this experiment).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load and preprocess the dataset
# NOTE: keras.datasets.mnist.load_data() downloads the real MNIST dataset
# from Google's servers. In a normal environment (with internet access)
# simply use:
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# A small synthetic fallback (28x28 blob images) is provided below only so
# this script still runs end-to-end in network-restricted sandboxes.
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
except Exception as e:
    print(f"Could not download real MNIST ({e}). Using a synthetic fallback dataset instead.")

    def make_synthetic_mnist(n, seed):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 10, n)
        X = np.zeros((n, 28, 28), dtype=np.float32)
        for i, digit in enumerate(y):
            cx, cy = 14 + rng.normal(0, 2), 14 + rng.normal(0, 2)
            radius = 6 + digit  # digit-dependent blob size so classes are learnable
            yy, xx = np.mgrid[0:28, 0:28]
            blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius)))
            X[i] = (blob * 255 + rng.normal(0, 10, (28, 28))).clip(0, 255)
        return X.astype(np.uint8), y.astype(np.uint8)

    x_train, y_train = make_synthetic_mnist(2000, seed=1)
    x_test, y_test = make_synthetic_mnist(400, seed=2)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# 2. Normalize pixel values to [0, 1] and reshape for CNN input (add channel dim)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# 3. Build CNN architecture using Keras
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 6. Display model summary
model.summary()

# 4. Train the model for 5 epochs
# (Using a subset for a quick lab-time demo; use full data for real training)
history = model.fit(
    x_train, y_train_cat,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=2,
)

# 5. Evaluate the model on the testing dataset
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Predict the class of one test image
sample = x_test[0:1]
pred_probs = model.predict(sample, verbose=0)
pred_class = np.argmax(pred_probs)
print(f"\nTrue label of test image #0: {y_test[0]}")
print(f"Predicted class: {pred_class}")
print(f"Predicted probabilities: {np.round(pred_probs[0], 4)}")
