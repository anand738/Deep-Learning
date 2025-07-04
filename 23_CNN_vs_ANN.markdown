# Tutorial 23 - Operation of CNN (CNN vs. ANN)

## 1. Overview of CNNs and ANNs
Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs) are both neural network architectures but are designed for different types of data and tasks.

- **Artificial Neural Network (ANN)**:
  - Uses fully connected (dense) layers.
  - Suitable for tabular data (e.g., spreadsheets) or non-spatial data.
  - Processes inputs as flat vectors, losing any spatial structure.
- **Convolutional Neural Network (CNN)**:
  - Specialized for grid-like data, such as images (2D) or time-series (1D).
  - Uses convolution, pooling, and fully connected layers to process data while preserving spatial relationships.

**Visual Description**: 
- A diagram comparing:
  - **ANN**: A flat input vector feeding into dense layers.
  - **CNN**: A 2D image processed through convolutional and pooling layers, followed by dense layers for classification.

  ![ANN VS CNN](src\ann_vs_cnn.webp)

## 2. CNN Operations
CNNs process images through a series of operations to extract and learn features efficiently:

- **Convolution**: 
  - Applies filters (e.g., 3x3 matrices) to the input image to extract features like edges or textures.
  - Produces feature maps that highlight patterns (see Tutorial 21).
- **Pooling**: 
  - Reduces the spatial size of feature maps (e.g., max pooling takes the maximum value in a region).
  - Decreases computational load and helps generalize by reducing overfitting.
- **Fully Connected Layers**: 
  - After convolution and pooling, flattened feature maps are fed into dense layers for tasks like classification or regression.
- **Advantages**:
  - **Parameter Efficiency**: Filters share weights across the image, reducing the number of parameters.
  - **Local Connectivity**: Focuses on local patterns (e.g., nearby pixels).
  - **Translation Invariance**: Detects features regardless of their position in the image.

**Visual Description**: 
- A diagram showing the CNN pipeline: 
  - Input image (e.g., 28x28) → Convolution (produces feature maps) → Pooling (reduces size) → Flattened → Dense layers → Output (e.g., class probabilities).
  - Each stage highlights the transformation (e.g., feature maps shrinking after pooling).

## 3. CNN vs. ANN
CNNs and ANNs differ significantly in how they handle data and perform on image-related tasks.

- **Input Handling**:
  - **ANN**: Flattens images into 1D vectors (e.g., a 28x28 image becomes 784 inputs), losing spatial relationships between pixels.
  - **CNN**: Preserves the 2D structure of images, processing local patches with filters to capture spatial patterns.
- **Parameters**:
  - **ANN**: Requires many weights due to full connectivity (e.g., 784 inputs to 128 neurons = 784 × 128 = 100,352 weights for one layer).
  - **CNN**: Uses fewer weights due to shared filters (e.g., a 3x3 filter = 9 weights per feature map, plus a bias).
- **Performance**:
  - **ANN**: Poor for image tasks due to high parameter count, leading to overfitting and inefficiency.
  - **CNN**: Excels in image tasks (e.g., object recognition) due to efficient feature extraction and better generalization.

**Visual Description**: 
- A table comparing ANN (dense layers, high parameter count) vs. CNN (convolution, pooling, fewer parameters).
- An example showing ANN and CNN performance on the MNIST dataset (e.g., CNN achieving higher accuracy with fewer parameters).

**Comparison Table**:

| **Aspect** | **ANN** | **CNN** |
|------------|---------|---------|
| **Input Handling** | Flattens images (e.g., 28x28 → 784) | Preserves 2D structure |
| **Parameters** | High (e.g., 784 × 128 = 100,352 weights) | Low (e.g., 3x3 filter = 9 weights) |
| **Performance** | Poor for images, prone to overfitting | Excels in image tasks, generalizes better |
| **Use Case** | Tabular data, non-spatial tasks | Images, grid-like data (e.g., MNIST, CIFAR-10) |
| **Pros** | Simple, versatile for non-spatial data | Parameter-efficient, captures spatial patterns |
| **Cons** | Inefficient for images, high memory use | More complex to design, task-specific |

## 4. Code Example
Below is a Python code example using Keras to compare an ANN and a CNN on dummy image data for a binary classification task.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np

# Dummy data: 4 images of size 28x28x1 (grayscale)
X = np.random.rand(4, 28, 28, 1)
y = np.array([0, 1, 0, 1])  # Binary labels

# ANN Model
ann = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X, y, epochs=10, verbose=0)

# CNN Model
cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X, y, epochs=10, verbose=0)

# Predict on one image
pred_ann = ann.predict(X[:1])
pred_cnn = cnn.predict(X[:1])
print(f"ANN Prediction: {pred_ann[0][0]:.2f}")
print(f"CNN Prediction: {pred_cnn[0][0]:.2f}")
```

**Output** (example, varies due to random data):
```
ANN Prediction: 0.45
CNN Prediction: 0.52
```

**Explanation**:
- **ANN**: Flattens the 28x28x1 image into a 784-dimensional vector, feeding it into dense layers (128 neurons, then 1 output). This requires many parameters (e.g., 784 × 128 + 128 = 100,480 weights for the first layer).
- **CNN**: Applies 16 filters (3x3) and max pooling (2x2), reducing spatial dimensions (28x28 → 13x13 after pooling), then flattens to a dense layer. Fewer parameters (e.g., 16 × (3x3 + 1) = 160 weights for convolution).
- The CNN preserves spatial structure, making it more efficient for images.