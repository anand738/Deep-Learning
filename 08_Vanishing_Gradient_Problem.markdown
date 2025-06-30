# Notes on Tutorial 8 - Vanishing Gradient Problem

## 1. What Is the Vanishing Gradient Problem?
The vanishing gradient problem occurs when gradients in backpropagation become very small, slowing or stopping learning in early layers of deep networks.

- **Cause**: Gradients are multiplied through layers (chain rule). Sigmoid/Tanh activations produce small derivatives (e.g., Sigmoid: \( f'(z) \leq 0.25 \)).
- **Impact**: Early layers’ weights barely update, leading to poor performance.

**Visual Description**: Neural network with gradients shrinking backward. Graph of gradient magnitude vs. layer, near zero for early layers.

## 2. Why It Happens
- **Activation Functions**:
  - **Sigmoid**: 1 / (1 + e⁻ᶻ), derivative σ'(z) = σ(z)(1-σ(z)) \ leq 0.25.
  - **Tanh**: tanh(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ), derivative 1 - tanh²(z) \leq 1.
  - Small derivatives multiply to tiny values in deep networks.
- **Deep Architectures**: More layers amplify gradient shrinkage.


## 3. Solutions
- **ReLU Activation**: f(z) = max(0, z), derivative 1 for z > 0 , avoids vanishing gradients.
- **Leaky ReLU**: f(z) = max(alpha z, z), alpha = 0.01 , prevents dying ReLU.
- **Weight Initialization**: Xavier (for Sigmoid/Tanh) or He (for ReLU) balances gradients.
- **Batch Normalization**: Normalizes layer outputs, stabilizes gradient flow.
- **Skip Connections**: ResNets allow gradients to bypass layers.


## 4. Code Example
**Code**: Comparing Sigmoid vs. ReLU to mitigate vanishing gradients in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sample data: 4 samples, 2 features (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Sigmoid model (prone to vanishing gradients)
model_sigmoid = Sequential([
    Dense(8, input_dim=2, activation='sigmoid'),
    Dense(8, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])
model_sigmoid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_sigmoid.fit(X, y, epochs=200, verbose=0)

# ReLU model
model_relu = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_relu.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_relu.fit(X, y, epochs=200, verbose=0)

# Predict
pred_sigmoid = model_sigmoid.predict(np.array([[1, 0]]))
pred_relu = model_relu.predict(np.array([[1, 0]]))
print(f"Sigmoid Prediction for [1, 0]: {pred_sigmoid[0][0]:.2f}")
print(f"ReLU Prediction for [1, 0]: {pred_relu[0][0]:.2f}")
```
*Output*: ReLU model converges faster, predicts closer to 1 for [1, 0].

**Explanation**: ReLU avoids vanishing gradients, improving training.

