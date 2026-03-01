# CIFAR-10 Image Classification using CNN

## Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset into ten categories. The objective is to build a complete deep learning pipeline including data preprocessing, augmentation, model training, evaluation, and visualization of learned features.

The final trained model achieved a **test accuracy of 81.3%**.

---

## Dataset

The CIFAR-10 dataset contains:

- 60,000 color images (32x32 pixels)
- 10 classes:
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck
- 50,000 training images
- 10,000 test images

The dataset is loaded directly using:

```python
from tensorflow.keras.datasets import cifar10
```

## CNN Architecture

The model consists of three convolutional blocks:

1. Conv2D with 32 filters, 3x3 kernel, ReLU activation
2. Conv2D with 64 filters, 3x3 kernel, ReLU activation
3. Conv2D with 128 filters, 3x3 kernel, ReLU activation

Each block is followed by a MaxPooling layer. A dropout layer (rate 0.5) is applied to reduce overfitting. After the convolutional blocks, a fully connected Dense layer with 128 units and ReLU activation is added, and the output layer uses 10 units with Softmax activation for classification.

### Design Justification

- Increasing filters (32 → 64 → 128) allows progressive feature learning.
- 3x3 kernels capture local spatial patterns effectively.
- MaxPooling reduces spatial dimensions and computational cost.
- Dropout prevents overfitting.
- Softmax provides a probability distribution across the ten classes.

## Data Preprocessing

- Pixel values are normalized to the range [0, 1].
- Labels are converted to one-hot encoding.
- Training data is augmented using random rotations, horizontal flips, and width/height shifts.

Data augmentation improves generalization and helps reduce overfitting.

## Training Strategy

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 50
- Batch Size: 64
- Learning Rate Scheduler: ReduceLROnPlateau
- Model Checkpoint: Saves the model with the best validation accuracy

## Model Performance

- Final test accuracy: **81.3%**

### Classification Report Summary

**Strong performance on:**

- Automobile
- Ship
- Truck
- Frog

**Most confusion observed between:**

- Cat and Dog
- Bird and Airplane

## Visualizations

The project includes the following visual outputs:

- Training & validation accuracy/loss curves
- Confusion matrix
- Examples of misclassified images
- Feature map visualizations from convolutional layers

All key outputs are stored in the `outputs/` directory.

## Project Structure

```
CNN-CIFAR10/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── outputs/
│   ├── accuracy_loss_plot.png
│   ├── confusion_matrix.png
│   └── feature_maps.png
```

## How to Run the Project

1. Clone the repository:

```bash
git clone <https://github.com/ashifa-1/CNN-CIFAR10>
cd CNN-CIFAR10
```

2. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the notebook:

   Open `notebook.ipynb` in Jupyter or VS Code and execute all cells from top to bottom.

## Conclusion

This project demonstrates a complete deep learning workflow for image classification using convolutional neural networks. The model achieves over 80% test accuracy and provides insights into feature extraction, the impact of data augmentation, evaluation metrics, and visualization of learned representations.

## Author

Ashifa Mohammed Ahmad
