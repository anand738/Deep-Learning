# Session on Loss Functions in Deep Learning

## Overview
Loss functions measure the discrepancy between a model's predictions and actual targets, guiding optimization in deep learning. The choice of loss function depends on the task, data characteristics, and model output. This session covers common loss functions, their formulations, use cases, advantages, and limitations.

## Categories of Loss Functions
1. **Regression Loss Functions**: For continuous output predictions.
2. **Classification Loss Functions**: For discrete class predictions.
3. **Specialized Loss Functions**: For tasks like segmentation or generative modeling.

## 1. Regression Loss Functions
### Mean Squared Error (MSE)
- **Formula**: $ L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $
- **Use Case**: Predicting continuous values (e.g., house prices, temperature).
- **Pros**: 
  - Simple, differentiable.
  - Emphasizes large errors.
- **Cons**: 
  - Sensitive to outliers.
  - Assumes Gaussian error distribution.

### Mean Absolute Error (MAE)
- **Formula**: $ L = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| $
- **Use Case**: Regression with outliers (e.g., financial forecasting).
- **Pros**: 
  - Robust to outliers.
  - Intuitive (average absolute error).
- **Cons**: 
  - Non-differentiable at zero.
  - Less focus on large errors.

### Huber Loss
- **Formula**: 
  - If $|y_i - \hat{y}_i| \leq \delta$: $ L = \frac{1}{2} (y_i - \hat{y}_i)^2 $
  - If $|y_i - \hat{y}_i| > \delta$: $ L = \delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2 $
- **Use Case**: Regression balancing robustness and sensitivity (e.g., robust regression).
- **Pros**: 
  - Combines MSE and MAE benefits.
  - Less outlier-sensitive than MSE.
- **Cons**: 
  - Requires tuning $\delta$.
  - More complex than MSE/MAE.

## 2. Classification Loss Functions
### Binary Cross-Entropy (Log Loss)
- **Formula**: $ L = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)] $
- **Use Case**: Binary classification (e.g., spam detection).
- **Pros**: 
  - Probabilistic, encourages confident predictions.
  - Works with sigmoid outputs.
- **Cons**: 
  - Sensitive to class imbalance.
  - Unstable with extreme predictions.

### Categorical Cross-Entropy
- **Formula**: $ L = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c}) $
- **Use Case**: Multi-class classification (e.g., image classification).
- **Pros**: 
  - Effective for multi-class tasks.
  - Compatible with softmax outputs.
- **Cons**: 
  - Sensitive to class imbalance.
  - Assumes exclusive classes.

### Hinge Loss
- **Formula**: $ L = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i \cdot \hat{y}_i) $
- **Use Case**: Binary classification, SVMs (e.g., text classification).
- **Pros**: 
  - Promotes margin maximization.
  - Robust to outliers.
- **Cons**: 
  - Not probabilistic.
  - Non-differentiable at certain points.

## 3. Specialized Loss Functions
### Dice Loss
- **Formula**: $ L = 1 - \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} $
- **Use Case**: Image segmentation (e.g., medical imaging).
- **Pros**: 
  - Handles class imbalance well.
  - Focuses on region overlap.
- **Cons**: 
  - Unstable for small objects.
  - Less suited for non-segmentation tasks.

### Kullback-Leibler (KL) Divergence
- **Formula**: $ L = \sum_{i=1}^n y_i \log\left(\frac{y_i}{\hat{y}_i}\right) $
- **Use Case**: Generative models (e.g., variational autoencoders).
- **Pros**: 
  - Measures distribution divergence.
  - Useful for probabilistic modeling.
- **Cons**: 
  - Asymmetric, sensitive to small probabilities.
  - Numerically unstable.

## Comparison Table
| **Loss Function** | **Task** | **Speed** | **Stability** | **Use Case** | **Pros** | **Cons** |
|-------------------|----------|-----------|---------------|--------------|----------|----------|
| **MSE** | Regression | Fast | Moderate | Continuous prediction | Simple, emphasizes large errors | Outlier-sensitive |
| **MAE** | Regression | Moderate | High | Regression with outliers | Robust to outliers | Non-differentiable at zero |
| **Huber Loss** | Regression | Moderate | High | Robust regression | Balances MSE and MAE | Needs $\delta$ tuning |
| **Binary Cross-Entropy** | Binary Classification | Fast | Moderate | Binary classification | Probabilistic, confident predictions | Class imbalance issues |
| **Categorical Cross-Entropy** | Multi-class Classification | Fast | Moderate | Multi-class classification | Works with softmax | Class imbalance issues |
| **Hinge Loss** | Binary Classification | Moderate | High | SVMs, binary tasks | Margin maximization | Not probabilistic |
| **Dice Loss** | Segmentation | Moderate | Moderate | Image segmentation | Handles imbalance | Unstable for small objects |
| **KL Divergence** | Generative Models | Moderate | Low | Generative tasks | Measures distribution difference | Numerically unstable |

## Key Considerations
- **Task Alignment**: Choose loss functions based on the task (e.g., regression, classification).
- **Data Properties**: Account for outliers, class imbalance, or distribution characteristics.
- **Model Compatibility**: Ensure the loss matches the model’s output (e.g., softmax for categorical cross-entropy).
- **Tuning**: Some losses (e.g., Huber) require hyperparameter tuning for optimal performance.