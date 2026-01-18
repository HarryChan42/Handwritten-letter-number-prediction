# Handwritten-number-prediction

This project implements an end-to-end handwritten digit recognition system using a Convolutional Neural Network (CNN) trained on the MNIST dataset.  
It also includes an interactive desktop drawing pad that allows users to draw digits with a mouse and receive real-time predictions with confidence scores.

The project demonstrates the full machine learning workflow: **data preprocessing → model training → evaluation → deployment in a GUI application**.

---

## Project Overview

The software is composed of two main parts:

1. **Model Training and Evaluation**
   - Train a CNN on the MNIST dataset using TensorFlow/Keras
   - Evaluate performance using accuracy, confusion matrix, and misclassification visualisation
   - Save the trained model for reuse

2. **Interactive Drawing Pad**
   - Desktop GUI built with Tkinter
   - Users draw digits directly on a canvas
   - Custom preprocessing converts drawings into MNIST-compatible inputs
   - The trained CNN predicts the digit and outputs a confidence score

---

## Features

- CNN-based handwritten digit classification
- MNIST dataset preprocessing and normalisation
- Validation during training
- Confusion matrix visualisation
- Misclassified image inspection
- Interactive drawing pad with mouse input
- Confidence score display for predictions
- Model saving and loading using `.keras` format

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Tkinter (GUI)

---
## Dataset

This project uses the **MNIST handwritten digits dataset**


- 60,000 training images
- 10,000 test images
- Image size: 28×28 pixels
- Grayscale images of digits from 0 to 9

The dataset is loaded directly via:


Model Architecture

The CNN architecture consists of:

Input layer: 28×28×1 grayscale image

Two convolutional layers with ReLU activation

Max pooling layers for downsampling

Fully connected dense layer for feature learning

Dropout layer for regularisation

Softmax output layer for 10-class classification

Architecture Summary

Conv2D (32 filters, 3×3)

MaxPooling2D

Conv2D (64 filters, 3×3)

MaxPooling2D

Flatten

Dense (128 units, ReLU)

Dropout (0.3)

Dense (10 units, Softmax)

Loss Function: sparse_categorical_crossentropy
Optimiser: Adam Optimiser

## Setup and activation

1. Clone the repository
- git clone https://github.com/your-username/mnist-digit-pad-cnn.git
cd mnist-digit-pad-cnn
2. Create and activate a virtual environment

- python -m venv venv
venv\Scripts\activate
3. Install dependencies
- pip install -r requirements.txt
  
# How to use

Draw a digit (0–9) using the mouse

Click Predict

The application displays:

Predicted digit

Confidence score

Click Clear to reset the canvas

## Notes and Limitations

Hand-drawn digits may differ from MNIST samples, which can affect accuracy

Stroke thickness and centering influence predictions

The application runs on CPU only

Tkinter must be available (included with most Python installations on Windows)

## FWhat's New / key Modifications

Model & Training

Upgraded from MNIST digit-only classification to EMNIST ByClass, supporting:

Digits (0–9)

Uppercase letters (A–Z)

Lowercase letters (a–z)

62 classes in total

Added EMNIST-specific orientation correction (rotate + flip) during preprocessing

Introduced light data augmentation (translation, rotation, zoom) to improve robustness

Implemented validation split from training data

Added training stabilisation using:

Early stopping (best weights restored)

Learning rate reduction on plateau

Expanded evaluation with:

Full 62×62 confusion matrix

Detailed classification report (precision, recall, F1-score)

Identification of most frequently confused class pairs

Exported reusable inference assets:

Trained .keras model

class_names.txt for class index → character mapping

Drawing Pad & Inference

Extended drawing pad to support digits and letters, not just numbers

Added live prediction loop (automatic inference every ~150 ms)

Implemented Top-K probability display for better interpretability

Added ink centering and scaling to better match EMNIST data distribution

Introduced 28×28 preview panel showing the exact input fed to the model

Added configurable preprocessing toggles:

Optional color inversion

Optional rotation and horizontal flip to match training orientation

Improved UX with:

Adjustable brush size

Confidence score display

Clean separation between drawing, preview, and prediction panels

Engineering & Reproducibility

Clear separation between:

Training / evaluation (train_emnist_byclass.py)

Inference / GUI (draw_pad_predict_emnist.py)

Model and class-label mapping saved explicitly for deployment

Code structured for easy extension (e.g. EMNIST Letters-only, MNIST fallback)

Designed to run out-of-the-box on CPU with minimal dependencies

## License

This project is licensed under the MIT License.

**Author:** Hoi Bong Chan
**Language:** C++  
**Frameworks:** Pycharm, Python, Tensorflow
**Keywords:** Machine Learning, PyCharm, Python, Tensorflow, Numpy, tkinter, keras
