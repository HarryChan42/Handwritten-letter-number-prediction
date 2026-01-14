import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Preprocessing
X_train = (X_train.astype("float32") / 255.0)[..., np.newaxis]  # (60000, 28, 28, 1)
X_test  = (X_test.astype("float32") / 255.0)[..., np.newaxis]   # (10000, 28, 28, 1)
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#Model Train
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=1
)

# Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Confusion matrix
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("MNIST Confusion Matrix")
plt.show()

# Missclassification exeption throw
wrong_idx = np.where(y_pred != y_test)[0]

if len(wrong_idx) > 0:
    n_show = 12
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(wrong_idx[:n_show]):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No misclassifications found (unlikely).")

model.predict(X_test[:1])
model.save("mnist_cnn.keras")
print("Saved as mnist_cnn.keras")