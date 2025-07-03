# Notes on Tutorial 12 -Gradient Descent

## 1. Overview of Optimization
Gradient Descent (GD) and Stochastic Gradient Descent (SGD) optimize neural network weights by minimizing the loss function.

- **GD**: Uses entire dataset to compute gradients.
- **SGD**: Uses one sample (or small batch) per update.



## 2. Gradient Descent
- **Process**:
  - Compute loss over all samples.
  - Calculate gradients: $   \frac{\partial L}{\partial w}   $.
  - Update: $ w = w - \eta \cdot \frac{\partial L}{\partial w} $.
- **Pros**: Stable, accurate gradients.
- **Cons**: Slow, memory-intensive for large datasets.

**Visual Description**: Smooth descent on a loss surface.
![Batch Gradient Decent](src\Batch_Gradient_Descent.webp)

## 3. Stochastic Gradient Descent
- **Process**:
  - Compute gradients for one sample (or mini-batch).
  - Update weights per sample/batch.
- **Pros**: Faster, escapes local minima due to noise.
- **Cons**: Noisy updates, may oscillate.

![SGD](src\Stochastic_Gradient_Descent.webp)


## 4. Mini-Batch GD
- Uses small batches (e.g., 32 samples).
- Balances speed and stability.

![Mini Batch Gradient](src\Mini_Batch_Gradient_Descent.webp)

**Visual Description**: Table comparing GD (stable, slow), SGD (fast, noisy), Mini-Batch GD (balanced).
![Comparision Tabel](src\comparision_of_GDs.png)

## 5. Code Example
**Code**: Comparing GD and SGD in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# Sample data: 4 samples, 2 features (OR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile with SGD
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train with SGD (batch_size=1)
model.fit(X, y, epochs=200, batch_size=1, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"SGD Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0] ≈ 1 (OR gate output).

**Explanation**: SGD (batch_size=1) is faster but noisier than GD (batch_size=4).

**Visual Description**: Loss curves
![Convergence](src\convergence.png)

