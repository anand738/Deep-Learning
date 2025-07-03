# Notes on Tutorial 12 - Various Weight Initialization Techniques in Neural Network

## 1. Importance of Weight Initialization
Initial weights affect gradient flow and training convergence. Poor initialization causes vanishing/exploding gradients.

- **Goal**: Start with weights that balance gradient magnitudes.
- **Impact**: Faster convergence, better performance.

**Visual Description**: Loss surface with paths for poor (divergent) vs. good (converging) initialization.

## 2. Initialization Techniques
### 2.1. Zero Initialization
- Weights set to 0.
- **Problem**: All neurons learn identical features, no diversity.
- **Use**: Rarely used.

### 2.2. Random Initialization
- Weights drawn from uniform/normal distribution (e.g., [-0.1, 0.1]).
- **Problem**: Large values cause exploding gradients; small values cause vanishing gradients.

### 2.3. Xavier/Glorot Initialization
- For Sigmoid/Tanh: \( W \sim \text{Normal}(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}) \).
- Balances variance for stable gradients.

### 2.4. He Initialization
- For ReLU: \( W \sim \text{Normal}(0, \sqrt{\frac{2}{n_{\text{in}}}}) \).
- Accounts for ReLU’s positive outputs, prevents vanishing gradients.

**Visual Description**: Weight distribution plots: Zero (spike at 0), Random (wide spread), Xavier/He (balanced). Loss curves for each.

## 3. Code Example
**Code**: Neural network with He initialization in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
import numpy as np

# Sample data: 4 samples, 2 features (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu', kernel_initializer=HeNormal()),
    Dense(8, activation='relu', kernel_initializer=HeNormal()),
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

**Explanation**: He initialization stabilizes training for ReLU layers.

**Visual Description**: Network diagram: 2 inputs → 8 hidden (ReLU, He init) → 8 hidden → 1 output. Loss curves for random vs. He initialization.

## 4. Next Steps
- Experiment with Xavier/He initialization.
- Test on Kaggle datasets.

**Visual Description**: Screenshot of Keras code or Kaggle notebook.