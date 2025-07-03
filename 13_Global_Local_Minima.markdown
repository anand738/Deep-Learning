# Notes on Tutorial 14 - Global Minima and Local Minima in Depth Understanding

## 1. What Are Minima?
- **Loss Function**: Measures error, forming a loss surface.
- **Global Minimum**: Lowest loss point, optimal weights.
- **Local Minimum**: Suboptimal point lower than nearby points.


## 2. Challenges
- **Gradient Descent**: Follows gradients but may get stuck in local minima.
- **Deep Networks**: Complex loss surfaces with many local minima.
- **Impact**: Suboptimal models if stuck in local minima.


## 3. Strategies to Avoid Local Minima
- **SGD**: Noise from per-sample updates helps escape local minima.
- **Momentum**: Adds inertia to updates, smoothing the path.
- **Learning Rate Scheduling**: Adjusts \( \eta \) to explore the surface.
- **Random Restarts**: Multiple trainings with different initializations.

![Local and Global Minima/Maxima](Notes\src\local_global.png)

## 4. Code Example
**Code**: SGD with learning rate scheduling in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

# Sample data: 4 samples, 2 features
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Learning rate scheduler
def scheduler(epoch):
    return 0.01 * (1.0 / (1.0 + 0.1 * epoch))

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=1, callbacks=[LearningRateScheduler(scheduler)], verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0]

**Explanation**: Learning rate scheduling helps escape local minima.

**Visual Description**: Loss surface with SGD path escaping a local minimum. Learning rate vs. epochs plot.

![Learning Rate](Notes\src\learningrates.jpeg)