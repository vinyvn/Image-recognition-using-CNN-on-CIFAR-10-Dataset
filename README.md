# Image-recognition-using-CNN-on-CIFAR-10-Dataset

This repository contains a Convolutional Neural Network (CNN) model for classifying images in the CIFAR-10 dataset using Keras. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to train a CNN model to accurately classify these images into their respective categories.

###Dataset

The CIFAR-10 dataset is automatically downloaded and loaded using Keras. It consists of two sets:

####Training set: Used for training the model, consisting of 50,000 images.
####Test set: Used for evaluating the model's performance, consisting of 10,000 images.

###Model Architecture

Convolutional Layer 1: 32 filters, each of size (3, 3), ReLU activation.

Convolutional Layer 2: 32 filters, each of size (3, 3), ReLU activation.

Max Pooling Layer 1: Pooling with size (2, 2).

Dropout Layer 1: Dropout rate of 0.25 to prevent overfitting.

Convolutional Layer 3: 64 filters, each of size (3, 3), ReLU activation.

Convolutional Layer 4: 64 filters, each of size (3, 3), ReLU activation.

Max Pooling Layer 2: Pooling with size (2, 2).

Dropout Layer 2: Dropout rate of 0.25 to prevent overfitting.

Flatten Layer: Flattens the output for fully connected layers.

Dense Layer 1: 512 units, ReLU activation.

Dropout Layer 3: Dropout rate of 0.5 to prevent overfitting.

Dense Layer 2: 10 units with softmax activation for 10 classes.

###Model Training

The model is compiled with categorical cross-entropy loss and the Adam optimizer. It is then trained on the training dataset for 10 epochs with a batch size of 128. Validation data from the test set is used to monitor the model's performance during training.

###Model Evaluation

After training, the model's performance is evaluated on the test set, and the test accuracy is printed to the console.

###How to Use

To run this code, you need Python with Keras and NumPy installed. You can clone this repository and execute the code in your preferred Python environment. Ensure that you have a reliable internet connection for the initial download of the CIFAR-10 dataset.

Feel free to modify the model architecture, hyperparameters, or training settings to experiment and improve the model's performance on the CIFAR-10 datase

