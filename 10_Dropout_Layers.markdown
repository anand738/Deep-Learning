# Notes on Tutorial 10 - Dropout Layers in Multi Neural Network

## 1. What Are Dropout Layers?
Dropout is a regularization technique to prevent overfitting by randomly deactivating (setting to zero) a fraction of neurons during training.

- **Mechanism**: During each iteration, neurons are dropped with probability \( p \) (e.g., 0.3). At test time, all neurons are used, weights scaled by \( 1-p \).
- **Purpose**: Forces the network to learn redundant patterns, improving generalization.


## 2. Why Use Dropout?
- **Overfitting**: Deep networks memorize training data, performing poorly on test data.
- **Dropout**: Reduces reliance on specific neurons, mimics ensemble learning.

**Visual Description**: Graph of training vs. validation accuracy: without dropout (diverging), with dropout (closer curves).

![Graph of training vs. validation accuracy](Notes\src\dropout_comprision.webp)

## 3. Implementation
- Applied to hidden layers, typical dropout rate: 0.2–0.5.
- In Keras: `Dropout(rate)` layer.

**Visual Description**: Network diagram with dropout layers marked between dense layers.
![Dropout Regularization](Notes\src\Dropout.webp)

## 4. Code Example
**Code**: Neural network with dropout in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Sample data: 4 samples, 2 features (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Build model with dropout
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=4, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0] ≈ 1 (XOR gate output).

**Explanation**: Dropout (30%) prevents overfitting, improving robustness.
