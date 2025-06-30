# Notes on Tutorial 7 - Chain Rule of Differentiation with BackPropagation

## 1. Backpropagation Overview
Backpropagation trains neural networks by computing gradients of the loss function w.r.t. weights/biases and updating them to minimize loss.

- **Steps**:
  1. Forward pass: Compute predictions.
  2. Loss calculation.
  3. Backward pass: Compute gradients using chain rule.
  4. Update parameters.


## 2. Chain Rule in Backpropagation
The chain rule computes derivatives of composite functions, critical for backpropagation.

- **Formula**: y depends on u, and u depends on x,dy/dx = (dy/du) × (du/dx).
- **In Neural Networks**: Computes ∂L/∂w by multiplying gradients through layers:
  - Loss L depends on the predicted output ŷ
  - ŷ depends on the neuron's weighted sum z
  - z depends on the weight w.
  - ∂L/∂w = (∂L/∂ŷ) × (∂ŷ/∂z) × (∂z/∂w).

## 3. Backpropagation Process
- **Forward Pass**:
  - Compute: zₗ = Wₗ · aₗ₋₁ + bₗ, aₗ = f(zₗ).
  - Calculate loss (e.g., L = ½(y - ŷ)²).
- **Backward Pass**:
  - Output layer gradient: δ_L = (∂L/∂ŷ) × f'(z_L).
  - Hidden layers: δₗ = (Wₗ₊₁ᵀ · δₗ₊₁) × f'(zₗ).
  - Weight gradients: ∂L/∂Wₗ = δₗ · aₗ₋₁ᵀ.
  - Update: Wₗ = Wₗ - η·(∂L/∂Wₗ).

## Example:

  **Forward**
  - z = (weight × input) + bias  
      = (0.5 × 1.0) + 0.2  
      = 0.7  
  - ŷ = σ(z) = 1 / (1 + e^-z)  
        ≈ 1 / (1 + e^-0.7)  
        ≈ 0.668  

  **Backword**
  - ∂L/∂ŷ = 2 × (y - ŷ)  
          ≈ 2 × (1 - 0.668)  
          ≈ 0.664  

  **Calculate Activation Gradient**:
  - σ'(z) = σ(z) × (1 - σ(z))  
          ≈ 0.668 × (1 - 0.668)  
          ≈ 0.222  

  **Compute Error Signal (δ)**
  - δ = (∂L/∂ŷ) × σ'(z)  
      ≈ 0.664 × 0.222  
      ≈ 0.147  

  **Calculate Weight Gradient:**
  - ∂L/∂w = δ × input  
      = 0.147 × 1.0  
      = 0.147  

  **Update the Weight:**
  - w_new = w - (learning rate × ∂L/∂w)  
      = 0.5 - (0.1 × 0.147)  
      ≈ 0.485  
