# Notes on Tutorial 3 - Introduction to Neural Network and Deep Learning

## 1. What Are Neural Networks?
Neural networks are computational models inspired by the human brain, used for tasks like classification, regression, and pattern recognition.

- **Structure**:
  - **Input Layer**: Receives features (e.g., image pixels).
  - **Hidden Layers**: Process data with weights, biases, and activations.
  - **Output Layer**: Produces results (e.g., class probabilities).
- **Components**:
  - **Weights**: Adjust input importance wi).
  - **Biases**: Shift outputs (bi).
  - **Activation Functions**: Add non-linearity (e.g., ReLU).

## 2. What Is Deep Learning?
Deep learning uses neural networks with many hidden layers to model complex patterns, excelling in tasks like image and speech recognition.

- **Why Deep?**: More layers capture hierarchical features (e.g., edges → shapes → objects in images).
- **Requirements**: Large datasets, GPUs/TPUs.



## 3. How Neural Networks Work

  ## 1. Forward Propagation
  - **Weighted Sum:** `z = (w₁ × x₁) + ... + b`
  - **Activation:** `a = ReLU(z)`
  - **Output:** Prediction `ŷ`

  ## 2. Loss Calculation
  - **MSE:** `½ × (y - ŷ)²`  
  - Example: `y=1.0`, `ŷ=0.8` → `MSE=0.02`

  ## 3. Backpropagation
  - **Gradient:** `∂Loss/∂w = (ŷ - y) × x`
  - **Update Rule:** `w_new = w_old - (η × gradient)`


## 4. # Activation Functions

  ## Sigmoid (`σ`)
  - **Formula:** `1 / (1 + e⁻ᶻ)`
  - **Range:** `[0, 1]`
  - **Use Case:** Binary classification (output layer).

  ## ReLU
  - **Formula:** `max(0, z)`
  - **Range:** `[0, +∞)`
  - **Use Case:** Hidden layers (default choice).

  ## Tanh
  - **Formula:** `(eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)`
  - **Range:** `[-1, 1]`
  - **Use Case:** Hidden layers (RNNs, zero-centered).
- **Note:** `tanh` is a scaled version of `sigmoid`.


## 5. Tools and Resources
- **Libraries**: TensorFlow/Keras, PyTorch.
- **Hardware**: GPUs (NVIDIA), TPUs (Google).
- **Learn**: Coursera (Deep Learning Specialization), Kaggle tutorials.
