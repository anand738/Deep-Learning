# Notes on Tutorial 18 - Keras Tuner Hyperparameter Tuning - How To Select Hidden Layers And Number of Hidden Neurons In ANN

## 1. What Is Hyperparameter Tuning?
Hyperparameters (e.g., number of layers, neurons, learning rate) are set before training and significantly affect model performance. Keras Tuner automates finding optimal hyperparameters.

- **Goal**: Optimize ANN architecture (layers, neurons) for better accuracy/loss.
- **Keras Tuner**: A library to search hyperparameter combinations using algorithms like Random Search or Hyperband.


## 2. Key Hyperparameters in ANNs
- **Number of Hidden Layers**: Affects model complexity (e.g., 1–5 layers).
- **Number of Neurons**: Determines layer capacity (e.g., 8–128 neurons).
- **Activation Function**: ReLU, Sigmoid, etc.
- **Learning Rate**: Impacts convergence speed.
- **Others**: Dropout rate, optimizer type.



## 3. Keras Tuner Workflow
1. Define model-building function with tunable hyperparameters.
2. Specify search space (e.g., range of layers, neurons).
3. Run tuner to evaluate combinations.
4. Select best model based on validation performance.


## 4. Code Example
**Code**: Using Keras Tuner to tune layers and neurons for XOR classification.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_tuner import RandomSearch
import numpy as np

# Sample data: XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Define model-building function
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model
# Initialize tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='xor_tuning'
)

# Search
tuner.search(X, y, epochs=100, validation_data=(X, y), verbose=0)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
pred = best_model.predict(np.array([[1, 0]]))
print(f"Prediction for [1, 0]: {pred[0][0]:.2f}")
```

**Explanation**: Keras Tuner tests combinations of layers (2-20) and neurons (32–512), selecting the best based on validation accuracy.


## 5. Tips for Tuning
- Start with wide ranges (e.g., 8–128 neurons), then narrow.
- Use Random Search for small datasets, Hyperband for larger ones.
- Monitor overfitting with validation data.

