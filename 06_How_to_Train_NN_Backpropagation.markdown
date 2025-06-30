# Notes on Tutorial 6 - How to Train Neural Network with Backpropagation

## 1. Training Overview
Training a neural network involves optimizing weights and biases to minimize loss using backpropagation and gradient descent.

- **Steps**:
  1. Forward propagation: Compute predictions.
  2. Loss calculation: Measure error.
  3. Backpropagation: Compute gradients.
  4. Update weights/biases.


## 2. Forward Propagation
- Data moves through layers: input → hidden → output.
- Neuron computation: \( z = (w_i.x_i) + b, a = f(z).
- Example: Input x = [1, 0], weights w = [0.4, -0.3], bias b = 0.2, ReLU:
  - z = (0.4 × 1) + (-0.3 × 0) + 0.2
  -   = 0.4 + 0 + 0.2
  -   = 0.6


## 3. Loss Function
- Quantifies error:
  - **MSE**: = (1/n) × Σ (y - ŷ)² 
  - **Binary Cross-Entropy**: Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)].


## 4. Backpropagation
- Computes gradients of loss w.r.t. weights/biases using chain rule.
- Updates: w_new = w - η×(∂L/∂w).
- Propagates gradients backward: output → hidden → input layers.

## 5. Training Process
1. Initialize weights/biases (randomly).
2. For each epoch:
   - Forward pass.
   - Compute loss.
   - Backpropagate gradients.
   - Update parameters.
3. Repeat until loss converges.


## 6. Code Example
**Code**: Training a neural network with backpropagation in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sample data: 4 samples, 2 features (AND gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=4, verbose=0)

# Predict
pred = model.predict(np.array([[1, 1]]))
print(f"Prediction for [1, 1]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 1] ≈ 1 (AND gate output).

**Explanation**: Backpropagation (handled by Keras) updates weights to learn the AND pattern.

## 7. Next Steps
- Experiment with learning rates, epochs.
- Test on Kaggle datasets.

**Visual Description**: Screenshot of Keras training log or Kaggle notebook.