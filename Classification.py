"""
train_emnist_byclass.py

EMNIST ByClass (62 classes: 0-9, A-Z, a-z) CNN training + evaluation + confusion matrix.
- Uses TensorFlow Datasets (tfds)
- Fixes EMNIST orientation (rotate + flip)
- Saves model: emnist_byclass_cnn.keras
- Saves class names: class_names.txt
- Saves confusion matrix image: confusion_matrix.png

Run:
  pip install -U tensorflow tensorflow-datasets scikit-learn matplotlib numpy
  python train_emnist_byclass.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# -------------------------
# Config
# -------------------------
DATASET_NAME = "emnist/byclass"
MODEL_OUT = "emnist_byclass1_cnn.keras"
CLASS_NAMES_OUT = "class_names1.txt"
CM_OUT = "confusion_matrix1.png"

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1          # split train
SEED = 42

# Augmentation (light) – set False if you want simplest pipeline
USE_AUGMENTATION = True

# Mixed precision can speed on some GPUs; safe to leave off
USE_MIXED_PRECISION = True


# -------------------------
# Class names (62)
# EMNIST/byclass label order is: digits (0-9), uppercase (A-Z), lowercase (a-z)
# -------------------------
def build_class_names_62():
    digits = [str(i) for i in range(10)]
    upper = [chr(ord("A") + i) for i in range(26)]
    lower = [chr(ord("a") + i) for i in range(26)]
    return digits + upper + lower


# -------------------------
# Preprocessing
# -------------------------
def fix_emnist_orientation(image):
    """
    EMNIST images often appear rotated/transposed depending on pipeline.
    This correction is widely used:
      rotate 90° clockwise (k=3) then flip left-right
    """
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    return image


def preprocess(image, label, num_classes):
    # image: uint8 [0..255], shape (28,28)
    image = fix_emnist_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # (28,28,1)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def make_augmenter():
    # Keep it light so letters don't become unreadable.
    return tf.keras.Sequential(
        [
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.08),
        ],
        name="augmenter",
    )


# -------------------------
# Model
# -------------------------
def build_cnn(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = inputs
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs, name="emnist_byclass_cnn")
    return model

#Util
def report_gpu_status():
    gpus = tf.config.list_physical_devices("GPU")
    is_gpu_accelerated = len(gpus) > 0

    print(f"GPU accelerated: {is_gpu_accelerated}")
    if is_gpu_accelerated:
        print("GPUs detected:")
        for g in gpus:
            print(" -", g.name)

        # Recommended: don't allocate all VRAM at once
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print("Could not set memory growth:", e)

    return is_gpu_accelerated
def plot_history(history, out_prefix="training"):
    # Accuracy
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train acc")
    plt.plot(history.history.get("val_accuracy", []), label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_accuracy.png", dpi=200)
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train loss")
    plt.plot(history.history.get("val_loss", []), label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", dpi=200)
    plt.close()


def save_class_names(class_names, path):
    with open(path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

# -------------------------
# Main
# -------------------------
def main():
    if USE_MIXED_PRECISION:

        # Enable only if you know your GPU benefits; works on modern NVIDIA GPUs.
        if USE_MIXED_PRECISION and tf.config.list_physical_devices('GPU'):
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision enabled")
        else:
            print("Mixed precision disabled")

    # Load to num_classes
    (ds_train_full, ds_test), ds_info = tfds.load(
        DATASET_NAME,
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )
    #print
    num_classes = ds_info.features["label"].num_classes  # should be 62
    print("Dataset:", DATASET_NAME)
    print("Num classes:", num_classes)
    print("Train examples:", ds_info.splits["train"].num_examples)
    print("Test examples :", ds_info.splits["test"].num_examples)

    # Prepare class names
    class_names = build_class_names_62()
    if len(class_names) != num_classes:
        print("WARNING: class_names length != num_classes. Using generic indices.")
        class_names = [str(i) for i in range(num_classes)]
    save_class_names(class_names, CLASS_NAMES_OUT)
    print("Saved:", CLASS_NAMES_OUT)

    # Split train into train/val
    train_count = ds_info.splits["train"].num_examples
    val_count = int(train_count * VAL_SPLIT)
    train_count_eff = train_count - val_count

    ds_train = ds_train_full.take(train_count_eff)
    ds_val = ds_train_full.skip(train_count_eff).take(val_count)

    # Preprocess
    ds_train = ds_train.map(lambda x, y: preprocess(x, y, num_classes),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.map(lambda x, y: preprocess(x, y, num_classes),
                        num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_proc = ds_test.map(lambda x, y: preprocess(x, y, num_classes),
                               num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle/batch/cache/prefetch
    ds_train = (
        ds_train
        .shuffle(10_000, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test_proc = ds_test_proc.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Build model
    base_model = build_cnn(num_classes)

    if USE_AUGMENTATION:
        augmenter = make_augmenter()
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
        x = augmenter(inputs)
        outputs = base_model(x)
        model = models.Model(inputs, outputs, name="emnist_byclass_cnn_aug")
    else:
        model = base_model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Model Save
    model.save(MODEL_OUT)
    print("Saved model:", MODEL_OUT)

    # History plot
    plot_history(history, out_prefix="training")
    print("Saved: training_accuracy.png, training_loss.png")

    # Evaluate
    test_loss, test_acc = model.evaluate(ds_test_proc, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    y_true = []
    y_pred = []

    for batch_images, batch_labels in ds_test_proc:
        probs = model.predict(batch_images, verbose=0)#Predict
        preds = np.argmax(probs, axis=1)
        true = np.argmax(batch_labels.numpy(), axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(true.tolist())

    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # 62 labels
    plt.figure(figsize=(18, 18))
    disp.plot(include_values=False, xticks_rotation=90, cmap=None)  # don't force colors
    plt.title("EMNIST ByClass Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=250)
    plt.close()
    print("Saved:", CM_OUT)

    # Classification report (text)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved: classification_report.txt")

    # most confused pairs
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_idx = np.argsort(cm_off.ravel())[::-1]
    top_k = 15
    print("\nTop confusions (true -> predicted : count):")
    shown = 0
    for idx in flat_idx:
        if cm_off.ravel()[idx] == 0:
            break
        i, j = np.unravel_index(idx, cm_off.shape)
        print(f"  {class_names[i]} -> {class_names[j]} : {cm_off[i, j]}")
        shown += 1
        if shown >= top_k:
            break

#main run
if __name__ == "__main__":
    main()
