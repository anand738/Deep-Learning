# Tutorial 33 - Recurrent Neural Network Forward Propagation With Time

## 1. What Is RNN Forward Propagation?
Forward propagation in Recurrent Neural Networks (RNNs) is the process of computing outputs by passing input sequences through the network, incorporating temporal dependencies via recurrent connections. Unlike feedforward networks or CNNs, RNNs process sequences by maintaining a hidden state that evolves over time steps, capturing information from previous inputs.

- **Process**: At each time step, the RNN takes an input, updates its hidden state based on the current input and previous hidden state, and produces an output.
- **Purpose**: Enables RNNs to model sequential data (e.g., time-series, text) for tasks like sentiment analysis or video frame analysis.
- **Key Feature**: The recurrent loop allows information to persist across time steps, unlike the static processing in CNNs.

![Forward Prapogation](src\forward_prop_rnn.jpg)

## 2. Mechanics of RNN Forward Propagation
RNN forward propagation processes a sequence of inputs over time steps, updating the hidden state and generating outputs. Here’s how it works:

1. **Input Sequence**: A sequence of vectors $[x_1, x_2, \dots, x_T]$, where $x_t$ is the input at time step $t$ (e.g., a word embedding or time-series value).
2. **Hidden State Update**: At each time step $t$, the hidden state $h_t$ is computed as:
   $ h_t = \text{activation}(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $
   - $x_t$: Input at time $t$.
   - $h_{t-1}$: Previous hidden state (initially $h_0$, often set to zeros).
   - $W_{xh}$: Input-to-hidden weight matrix.
   - $W_{hh}$: Hidden-to-hidden weight matrix (recurrent connection).
   - $b_h$: Bias for hidden state.
   - $\text{activation}$: Typically tanh or ReLU.
3. **Output**: The output $y_t$ at time $t$ is computed as:
   $ y_t = W_{hy}h_t + b_y $
   - $W_{hy}$: Hidden-to-output weight matrix.
   - $b_y$: Output bias.
4. **Temporal Flow**: The hidden state $h_t$ carries information from all previous time steps, enabling the RNN to model sequential dependencies.

**Key Parameters**:
- **Input Size**: Dimension of $x_t$ (e.g., 32 for an embedding).
- **Hidden Size**: Dimension of $h_t$ (e.g., 64 units).
- **Output Size**: Dimension of $y_t$ (e.g., 1 for binary classification).
- **Sequence Length ($T$)**: Number of time steps (e.g., 10 words in a sentence).


## 3. Why Temporal Forward Propagation Matters
- **Sequential Dependencies**: The recurrent connection allows RNNs to capture relationships between elements in a sequence (e.g., word order in a sentence or frame transitions in a video).
- **Context Retention**: The hidden state acts as a memory, carrying context across time steps (e.g., remembering earlier words to predict sentiment).
- **Contrast with CNNs**: Unlike CNNs which process spatial data in a single pass, RNNs process sequences iteratively, making them ideal for tasks like video object tracking or time-series forecasting.


## 4. Code Example
Below is a TensorFlow code example demonstrating RNN forward propagation for a sequence classification task (e.g., sentiment analysis on a dummy dataset). The code manually implements the forward pass to illustrate the mechanics.

```python
import tensorflow as tf
import numpy as np

# Dummy dataset: 5 sequences, each of length 10, with 32-dimensional inputs
X = np.random.randn(5, 10, 32).astype(np.float32)  # [batch, time_steps, input_size]
y = np.array([1, 0, 1, 0, 1], dtype=np.float32)  # Binary labels (e.g., sentiment)

# RNN parameters
input_size = 32
hidden_size = 64
output_size = 1

# Initialize weights and biases
W_xh = tf.random.normal([input_size, hidden_size])
W_hh = tf.random.normal([hidden_size, hidden_size])
W_hy = tf.random.normal([hidden_size, output_size])
b_h = tf.zeros([hidden_size])
b_y = tf.zeros([output_size])

# Manual RNN forward propagation
def rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y):
    batch_size, time_steps, _ = X.shape
    h_t = tf.zeros([batch_size, hidden_size])  # Initial hidden state
    outputs = []
    
    for t in range(time_steps):
        x_t = X[:, t, :]  # Input at time step t
        h_t = tf.tanh(tf.matmul(x_t, W_xh) + tf.matmul(h_t, W_hh) + b_h)  # Hidden state update
        y_t = tf.matmul(h_t, W_hy) + b_y  # Output
        outputs.append(y_t)
    
    return tf.stack(outputs, axis=1)  # [batch, time_steps, output_size]

# Forward pass
outputs = rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y)
final_output = outputs[:, -1, :]  # Use last time step for classification
predictions = tf.sigmoid(final_output)  # Sigmoid for binary classification

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Sequence {i+1} predicted sentiment: {'Positive' if pred > 0.5 else 'Negative'} (Probability: {pred.numpy()[0]:.2f})")

# Simple training loop (for illustration)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for epoch in range(5):
    with tf.GradientTape() as tape:
        tape.watch([W_xh, W_hh, W_hy, b_h, b_y])
        outputs = rnn_forward(X, W_xh, W_hh, W_hy, b_h, b_y)
        final_output = outputs[:, -1, :]
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, tf.sigmoid(final_output)))
    grads = tape.gradient(loss, [W_xh, W_hh, W_hy, b_h, b_y])
    optimizer.apply_gradients(zip(grads, [W_xh, W_hh, W_hy, b_h, b_y]))
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
```

**Output** (example, varies due to random data):
```
Sequence 1 predicted sentiment: Positive (Probability: 0.62)
Sequence 2 predicted sentiment: Negative (Probability: 0.38)
...
Epoch 1, Loss: 0.6932
...
Epoch 5, Loss: 0.5901
```

**Explanation**:
- **Model**: A manual RNN implementation with a single layer, processing a sequence of 10 time steps with 32-dimensional inputs.
- **Input**: Dummy sequences (5 samples, 10 time steps, 32 features) and binary labels.
- **Forward Propagation**: Computes hidden states and outputs iteratively for each time step, using tanh activation and matrix operations.
- **Output**: The final time step’s output is used for binary classification (e.g., sentiment).
- **Training**: A simple gradient descent loop updates weights to minimize binary cross-entropy loss.


## 5. Benefits and Challenges
- **Benefits**:
  - **Temporal Modeling**: Captures sequential dependencies effectively (e.g., for video frame analysis or text).
  - **Compact Representation**: Hidden state summarizes past information efficiently.
  - **Flexibility**: Handles variable-length sequences.
- **Challenges**:
  - **Vanishing Gradients**: Long sequences can destabilize training (use LSTM/GRU, as in Tutorial 32).
  - **Computational Cost**: Iterative processing is slower than CNNs for large sequences.
  - **Complexity**: Manual implementation requires careful handling of time steps.

