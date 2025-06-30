# Notes on Tutorial 4 - How Does Neural Network Work

## 1. Neural Network Overview
Neural networks process data through layers of interconnected neurons to make predictions or decisions.

- **Layers**:
  - Input: Receives data (e.g., features).
  - Hidden: Transforms data with weights, biases, activations.
  - Output: Produces results (e.g., class labels).
- **Computation**: Each neuron calculates z = Σ (w_i · x_i) + b, then applies activation a = f(z).

**Visual Description**: Diagram of a neural network: 3 inputs → 2 hidden layers (4 neurons each) → 2 outputs. Equations at each neuron 

![Neural Network](src\Neural-Networks-Architecture.png)

## 2. Forward Propagation
- Data flows from input to output.
- Each layer computes:  z = (w_i · x_i) + b, a = f(z).
- Example:

- Inputs (x): [1, 2], Weights (w): [0.3, -0.2], Bias (b): 0.1

- Calculation:
  z = (0.3 × 1) + (-0.2 × 2) + 0.1
  = 0.3 - 0.4 + 0.1
  = 0.0



## 3. Loss Function
- Used for: Regression (predicting numbers, like house prices).

- Takes the average of squared differences between predicted (ŷ) and actual (y) values.

Bigger errors are punished more (because of squaring).
  - **Mean Squared Error (MSE)** = (1/n) × Σ (Actual - Predicted)²  
  - **Binary Cross-Entropy**: Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]  For binary classification.


## 4. Backpropagation
- Computes gradients of loss w.r.t. weights/biases using the chain rule.


- How It Works
  1. **Forward Pass:** Predict (`ŷ`), compute error (`y - ŷ`).  
  2. **Backward Pass:**  
    - Calculate gradients using **chain rule**.  
    - Propagate error backward.  
  3. **Update Weights:**  
    `w = w - (Learning Rate × Gradient)`  

- Key Concepts
  - **Chain Rule:** Breaks gradients into small steps.  
  - **Gradient Flow:** Error signals move backward.  
  - **Loss Surface:** 3D "valley" of error (goal: reach the bottom).  

- Example 
  - If `ŷ = 0.8`, `y = 1.0`, MSE = `0.04`.  
  - Gradient `∂Loss/∂ŷ = -0.4`.  
  - Adjust weights: `w = w - (η × gradient)`.  


![Prppagation](src\propogation.webp)

## 5. Training Process
1. Initialize weights/biases (randomly).
2. For each epoch:
   - Forward propagation.
   - Compute loss.
   - Backpropagate gradients.
   - Update weights/biases.
3. Repeat until convergence.


## 6. Code Example
**Code**: Neural network for binary classification in Keras.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sample data: 4 samples, 2 features (OR gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Build model
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=4, verbose=0)

# Predict
pred = model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```
*Output*: Prediction for [1, 0] ≈ 1 (OR gate output).

**Explanation**: The model learns the pattern via forward/backpropagation.

## 7. Next Steps
- Experiment with layers and neurons in Keras.
- Explore Kaggle datasets.