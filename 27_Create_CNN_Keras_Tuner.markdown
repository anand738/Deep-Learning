# Notes on Tutorial 27 - Create CNN Model and Optimize using Keras Tuner- Deep Learning

## 1. Overview of CNN Optimization
Convolutional Neural Networks (CNNs) require tuning hyperparameters like number of filters, kernel size, layers, and neurons to achieve optimal performance. Keras Tuner automates this process.

- **Goal**: Find the best CNN architecture for tasks like image classification.
- **Keras Tuner**: Searches hyperparameter combinations using Random Search, Hyperband, or Bayesian Optimization.

**Visual Description**: Diagram of a CNN with tunable components (e.g., filters, layers). Flowchart: define search space → evaluate models → select best.

## 2. Key CNN Hyperparameters
- **Number of Filters**: Filters in Conv2D layers (e.g., 16–64).
- **Kernel Size**: Filter size (e.g., 3x3, 5x5).
- **Number of Layers**: Conv and dense layers (e.g., 1–3 conv layers).
- **Neurons in Dense Layers**: Units in fully connected layers (e.g., 16–128).
- **Dropout Rate**: Regularization to prevent overfitting (e.g., 0.2–0.5).
- **Learning Rate**: Optimizer step size (e.g., 0.001–0.01).

**Visual Description**: Table of hyperparameters: filters, kernel size, layers, etc., with typical ranges. CNN diagram highlighting tunable parts.

## 3. Keras Tuner Workflow
1. Define a model-building function with tunable hyperparameters.
2. Specify search space (e.g., filters, layers).
3. Run tuner to evaluate combinations (e.g., Random Search).
4. Select best model based on validation accuracy/loss.

**Visual Description**: Code snippet screenshot of Keras Tuner setup. Plot of validation accuracy vs. hyperparameter combinations.

## 4. Code Example
**Code**: Optimizing a CNN for image classification using Keras Tuner.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_tuner import RandomSearch
import numpy as np

# Dummy data: 4 images (28x28x3, RGB)
X = np.random.rand(4, 28, 28, 3)
y = np.array([0, 1, 0, 1])

# Define model-building function
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('filters_1', 16, 64, step=16),
        kernel_size=hp.Choice('kernel_size_1', [3, 5]),
        activation='relu',
        input_shape=(28, 28, 3)
    ))
    model.add(MaxPooling2D((2, 2)))
    
    # Tune number of conv layers (1–2)
    for i in range(hp.Int('num_conv_layers', 1, 2)):
        model.add(Conv2D(
            filters=hp.Int(f'filters_{i+2}', 16, 64, step=16),
            kernel_size=hp.Choice(f'kernel_size_{i+2}', [3, 5]),
            activation='relu'
        ))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', 16, 64, step=16),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='cnn_tuning'
)

# Search
tuner.search(X, y, epochs=10, validation_data=(X, y), verbose=0)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
pred = best_model.predict(X[:1])
print(f"Prediction for first image: {pred[0][0]:.2f}")
```
*Output*: Prediction varies (random data).

**Explanation**: Keras Tuner optimizes filters (16–64), kernel sizes (3x3, 5x5), conv layers (1–2), dense units (16–64), and dropout (0.2–0.5), selecting the best configuration.

**Visual Description**: CNN diagram with tunable layers/filters. Bar chart of validation accuracy for top hyperparameter combinations.

## 5. Practical Tips
- **Search Space**: Start with wide ranges, then narrow based on results.
- **Trials**: Use 5–10 trials for small datasets, more for complex tasks.
- **Validation Data**: Ensure separate validation set to avoid overfitting.
- **Hyperband**: Use for faster tuning on large datasets.

**Visual Description**: Screenshot of Keras Tuner output in Colab. Plot of accuracy vs. trial number.

## 6. Next Steps
- Apply to Kaggle datasets (e.g., CIFAR-10).
- Tune additional parameters (e.g., learning rate).
- Explore Hyperband or Bayesian optimization.

**Visual Description**: Screenshot of Kaggle notebook or Keras Tuner logs.