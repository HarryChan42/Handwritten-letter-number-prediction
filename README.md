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

## Future Improvements

Improve preprocessing (centering by pixel mass, smoothing)

Data augmentation during training for better robustness

Display probability distribution for all digit classes

## License

This project is licensed under the MIT License.

**Author:** Hoi Bong Chan
**Language:** C++  
**Frameworks:** Pycharm, Python, Tensorflow
**Keywords:** Machine Learning, PyCharm, Python, Tensorflow, Numpy, tkinter, keras
