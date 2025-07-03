# Notes on Tutorial 15 - Types of Optimizer



## Overview of Optimizers
Optimizers minimize the loss function by updating neural network weights. Common optimizers:
- **SGD with Momentum**: Uses past gradients for smoother updates.
- **Adagrad**: Adapts learning rates per parameter.
- **RMSprop**: Uses moving average of squared gradients.
- **AdaDelta**: Extends RMSprop, no fixed learning rate.
- **Adam**: Combines momentum and RMSprop.

**Visual Description**: Table comparing optimizers: speed, stability, use case.


![Comparision Table](src\Optimizer_table.png)


## 1. What Is Adagrad?
Adagrad (Adaptive Gradient) adapts learning rates per parameter based on past gradients, effective for sparse data.
Adagrad is an abbreviation for Adaptive Gradient Algorithm. It is an adaptive learning rate optimization algorithm used for training deep learning models. It is particularly effective for sparse data or scenarios where features exhibit a large variation in magnitude.

Adagrad adjusts the learning rate for each parameter individually. Unlike standard gradient descent, where a fixed learning rate is applied to all parameters Adagrad adapts the learning rate based on the historical gradients for each parameter, allowing the model to focus on more important features and learn efficiently

- **Update Rule**:
  - Cache: $G_t = G_{t-1} + \left( \frac{\partial L}{\partial W} \right)^2$
  - Update: $W = W - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \frac{\partial L}{\partial W}$.
  - $ \epsilon $: Small constant (e.g., $ 10^{-8} $) to prevent division by zero.


### Benefits and Drawbacks
- **Pros**: No manual learning rate tuning, good for sparse data.
- **Cons**: Learning rate decays too fast, slowing convergence in deep networks.



## 2. Overview of AdaDelta

### **What is AdaDelta?**
AdaDelta is an extension of **AdaGrad** that adapts learning rates for each parameter while addressing its two main limitations:
1. **Aggressive learning rate decay** (AdaGrad's learning rates can become too small)
2. **Need to manually set an initial learning rate**

### **Key Features**
- Automatically adapts learning rates during training
- No need to set a base learning rate (unlike AdaGrad/RMSprop)
- Maintains per-parameter learning rates
- Combines ideas from **RMSprop** and **momentum**

### **Mathematical Formulation**

- **Adadelta**: An extension of Adagrad that addresses its aggressive, monotonically decreasing learning rate by using an exponentially decaying average of squared gradients and updates.

  - squared gradients: $ E[g^2]t = \rho E[g^2]{t-1} + (1 - \rho) \left( \frac{\partial L}{\partial W} \right)^2 $

  - squared parameter updates: $ E[\Delta W^2]t = \rho E[\Delta W^2]{t-1} + (1 - \rho) \Delta W_t^2 $

  - Update weights: $ W = W - \frac{\sqrt{E[\Delta W^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} \cdot \frac{\partial L}{\partial W}  $
  - Decay rate: $ \rho = 0.9 $
  - Small constant: $ \epsilon $ (added for numerical stability)



## 3. Overview of and RMSprop
- **RMSprop**: Improves Adagrad by using an exponentially moving average of squared gradients.
  - Update: $ E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) \left( \frac{\partial L}{\partial W} \right)^2 $ 
  - $W = W - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot \frac{\partial L}{\partial W}$.
  - Decay rate $ \rho = 0.9 $.

- **AdaDelta**: Extends RMSprop with a moving average of squared updates, no fixed learning rate.


### 3.1 Benefits
- **RMSprop**: Prevents premature convergence, good for deep networks.
- **AdaDelta**: No manual learning rate, robust for various tasks.





## 4 Overview of Adam Optimizer

  **Adam**: Combines momentum and RMSprop for adaptive learning rates and accelerated gradient descent.
  - Momentum update: $ v_t = \beta_1 v_{t-1} + (1 - \beta_1) \frac{\partial L}{\partial w} $
  - RMSprop update: $ g_t = \beta_2 g_{t-1} + (1 - \beta_2) \left( \frac{\partial L}{\partial w} \right)^2 $
  - Bias correction: 
  $ \hat{v}_t = \frac{v_t}{1 - \beta_1^t} $
  $ \hat{g}_t = \frac{g_t}{1 - \beta_2^t} $


  - Weight update: $ w = w - \eta \cdot \frac{\hat{v}_t}{\sqrt{\hat{g}_t} + \epsilon} $
  - Typical hyperparameters: 
  $ \beta_1 = 0.9 $ (momentum decay rate)
  $ \beta_2 = 0.999 $ (RMSprop decay rate)
  $ \epsilon = 10^{-8} $ (for numerical stability)


  - Pros: Fast convergence, robust for most tasks, handles sparse gradients well.
  - Cons: May overfit noisy data, sensitive to hyperparameter choices.



## 5. Code Example
**Code**: Neural network with Adam in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Sample data: 4 samples, 2 features (XOR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile with Adam
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=200, batch_size=4, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```


## 6. Choosing an Optimizer
- **SGD**: Simple, good with momentum.
- **Adagrad**: Sparse data.
- **RMSprop/AdaDelta**: Deep networks.
- **Adam**: Default for most tasks.

![Optimizers Comparision](src\optimizers.png)