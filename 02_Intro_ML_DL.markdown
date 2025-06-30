# Notes on Tutorial 2 - Introduction to Machine Learning and Deep Learning for Starters

## 1. What Is Machine Learning (ML)?
Machine learning is a field of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming.

- **Types**:
  - **Supervised Learning**: Uses labeled data (e.g., predict house prices). Includes regression (continuous outputs) and classification (discrete outputs).
  - **Unsupervised Learning**: Finds patterns in unlabeled data (e.g., clustering customers).
  - **Reinforcement Learning**: Learns via rewards (e.g., game-playing AI).


## 2. What Is Deep Learning (DL)?
Deep learning is a subset of ML using neural networks with multiple hidden layers to model complex, non-linear patterns.

- **Neural Networks**: Layers of interconnected nodes (neurons): input, hidden, output.
- **Why Deep?**: More layers capture intricate patterns (e.g., edges in images, semantics in text).
- **Requirements**: Large datasets, powerful hardware (GPUs).


## 3. How ML and DL Work
- **ML Workflow**:
  1. Collect data (e.g., house sizes, prices).
  2. Preprocess (clean, normalize).
  3. Train model (e.g., linear regression).
  4. Test on new data.
  5. Deploy model.
- **DL Workflow**:
  - Similar, but involves forward propagation (data through layers), loss calculation, backpropagation (gradient updates).



## 4. ML vs. DL
- **ML**: Simpler models (e.g., linear regression, SVM), less data, interpretable.
- **DL**: Complex models (neural networks), needs big data and compute, less interpretable.


## 5. Applications
- **ML**: Spam filters, recommendation systems, fraud detection.
- **DL**: Image classification (e.g., cats vs. dogs), NLP (chatbots), autonomous driving.

## 6. Code Example
**Code**: Linear regression (ML) vs. neural network (DL) in Keras.
```python
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Sample data: house sizes (sqft) and prices
X = np.array([[1400], [1600], [1700], [1875], [1100]])
y = np.array([245000, 312000, 279000, 308000, 199000])

# ML: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_pred = lr_model.predict([[1500]])
print(f"Linear Regression Prediction for 1500 sqft: ${lr_pred[0]:,.2f}")

# DL: Neural Network
nn_model = Sequential([
    Dense(8, input_dim=1, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X, y, epochs=100, verbose=0)
nn_pred = nn_model.predict([[1500]])
print(f"Neural Network Prediction for 1500 sqft: ${nn_pred[0][0]:,.2f}")
```
*Output*: Predictions for 1500 sqft house price (linear regression: ~$260K, neural network: similar).

**Explanation**: Linear regression is simple ML, while the neural network shows DL’s ability to model non-linear patterns (though overkill for this small dataset).


## 7. Next Steps
- **Learn**: Python, Scikit-learn for ML, Keras for DL.
- **Practice**: Kaggle datasets (e.g., Titanic).
- **Resources**: Coursera (Andrew Ng), fast.ai, X AI communities.
