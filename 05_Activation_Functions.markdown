# Notes on Tutorial 5 - Activation Functions

## 1. What Are Activation Functions?
Activation functions introduce non-linearity in neural networks, enabling them to model complex patterns.

- **Role**: Applied after neuron computation z = \sum (w_i \cdot x_i) + b to produce output \( a = f(z) \).
- **Need**: Without non-linearity, networks would be limited to linear transformations.

**Visual Description**: Diagram of a neuron: inputs → weighted sum \( z \) → activation \( f(z) \) → output. Linear vs. non-linear function graph.

## 2. Common Activation Functions
### 2.1. Sigmoid
- **Formula**: σ(z) = 1 / (1 + e⁻ᶻ).
- **Range**: 0 to 1.
- **Use**: Binary classification (output layer).
- **Pros**: Interpretable as probabilities.
- **Cons**: Vanishing gradients: derivative σ'(z) = σ(z)(1-σ(z)).

### 2.2. ReLU (Rectified Linear Unit)
- **Formula**: ReLU(z) = max(0, z).
- **Range**: 0 to ∞.
- **Use**: Hidden layers.
- **Pros**: Fast, avoids vanishing gradients.
- **Cons**: Dying ReLU (zero output for negative inputs).

### 2.3. Tanh
- **Formula**: tanh(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ).
- **Range**: -1 to 1.
- **Use**: Hidden layers, especially RNNs.
- **Pros**: Zero-centered, better for optimization.
- **Cons**: Vanishing gradients.

### 2.4. Softmax
- **Formula**: softmax(z_i) = e^{z_i} / ∑ e^{z_j}.
- **Range**: 0 to 1 (sums to 1).
- **Use**: Multi-class classification (output layer).
- **Pros**: Converts scores to probabilities.

### 2.5 Leaky Relu
- **Formula**:Leaky ReLU(z) = max(αz, z)
    (where α is typically 0.01)
- **Range**: (-∞, +∞)
- **Use Case**: Alternative to ReLU in hidden layers
- **Pros**: Solves "Dying ReLU" problem by allowing small negative outputs, Computationally efficient like ReLU
- **Cons**: Results may be inconsistent (α must be chosen), Not zero-centered

![Activation Function](Notes\src\activation.png)

## 3. Choosing Activation Functions
- **Hidden Layers**: ReLU (default), Tanh (if zero-centering needed).
- **Output Layer**:
  - Binary: Sigmoid.
  - Multi-class: Softmax.
  - Regression: None or Linear.


## 4. Code Example
**Code**: Neural network with ReLU and Sigmoid in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sample data: 4 samples, 2 features (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0] ≈ 1 (XOR gate output).

**Explanation**: ReLU enables non-linear learning in hidden layers, Sigmoid outputs binary probabilities.


## 5. Next Steps
- Experiment with Tanh, Softmax in Keras.
- Test on Kaggle datasets.
