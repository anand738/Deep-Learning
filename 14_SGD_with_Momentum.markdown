# Notes on Tutorial 14 - Stochastic Gradient Descent with Momentum

## 1. What Is SGD with Momentum?
SGD with momentum accelerates optimization by adding a fraction of the previous update to the current one, smoothing the path.

- **Update Rule**:
  - Velocity: $v_t = \gamma v_{t-1} + \eta \cdot \frac{\partial L}{\partial W}$
  - Weight update: $W = W - v_t$.
  - Momentum $ (\gamma) = 0.9$


## 2. Benefits
- **Faster Convergence**: Momentum accelerates toward minima.
- **Escapes Local Minima**: Inertia overcomes small dips.
- **Reduces Oscillations**: Smooths noisy updates.


## 3. Code Example
**Code**: SGD with momentum in Keras.
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

# Compile with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=200, batch_size=1, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0] ≈ 1 (OR gate output).

**Explanation**: Momentum smooths SGD updates, speeding convergence.

![SGD VS SGD with momentum](Notes\src\Accelrate_convergence.jpg)