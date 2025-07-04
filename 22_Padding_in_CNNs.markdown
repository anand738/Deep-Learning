# Tutorial 22 - Padding in Convolutional Neural Networks

## 1. What Is Padding in CNNs?
Padding involves adding borders, typically filled with zeros, around an input image before applying convolution. This controls the output size of the feature map and ensures that edge information is preserved during the convolution process.

- **Types of Padding**:
  - **Valid Padding**: No padding is added ($ p = 0 $). The output size is smaller than the input.
  - **Same Padding**: Enough padding is added to keep the output size equal to the input size (when stride = 1).
- **Purpose**:
  - Maintains spatial dimensions for consistent layer sizes.
  - Prevents loss of information at the image edges.
  - Enables deeper networks by avoiding excessive size reduction.

**Visual Description**: A diagram showing a 5x5 input image:
- **Valid Padding**: A 3x3 filter produces a 3x3 feature map (no border added, edges lose influence).
- **Same Padding**: A 1-pixel zero-filled border is added, yielding a 5x5 feature map after convolution with a 3x3 filter.

## 2. Why Use Padding?
Padding is critical in CNNs for several reasons:
- **Preserve Information**: Without padding, edge pixels are used less frequently in convolution, leading to information loss.
- **Control Output Size**: Same padding ensures the output size matches the input size, simplifying network design.
- **Enable Deeper Networks**: By preventing size reduction at each layer, padding allows stacking multiple convolutional layers without shrinking the feature map excessively.

**Visual Description**: An animation comparing convolution with padding

![Padding](src\padding.jpg)


## 3. Padding Mechanics
The output size of a convolution operation depends on the input size, filter size, padding, and stride. The general formula is:

$ \text{Output Size} = \lfloor \frac{n + 2p - f}{s} + 1 \rfloor $

Where:
- $ n $: Input dimension (width or height, assuming a square $ n \times n $ input).
- $ p $: Padding size (number of pixels added to each side).
- $ f $: Filter size (e.g., 3 for a 3x3 filter).
- $ s $: Stride (number of pixels the filter moves per step).
- $ \lfloor \cdot \rfloor $: Floor function (round down to the nearest integer).

### Valid Padding
- **Padding**: $ p = 0 $.
- **Output Size**: $ (n - f + 1) \times (n - f + 1) $.
  - For a 3x3 filter ($ f = 3 $), stride = 1: Output is $ (n - 2) \times (n - 2) $.
  - Example: 5x5 input → 3x3 output.

### Same Padding
- **Padding**: $ p = \lfloor (f - 1)/2 \rfloor $.
  - For a 3x3 filter ($ f = 3 $), $ p = \lfloor (3 - 1)/2 \rfloor = 1 $.
- **Output Size**: With $ s = 1 $, output is $ n \times n $.
  - Example: 5x5 input with $ p = 1 $ → 5x5 output.


## 4. Code Example
Below is a Python code example using Keras to demonstrate a CNN with **same padding**. The code applies a convolutional layer to a batch of images and shows how padding preserves the output size.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
import numpy as np

# Dummy data: 4 images of size 28x28x1 (grayscale)
X = np.random.rand(4, 28, 28, 1)

# Build CNN model with same padding
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))
])

# Apply convolution and get feature maps
feature_maps = model.predict(X)
print(f"Feature map shape: {feature_maps.shape}")
```

**Output**: 
```
Feature map shape: (4, 28, 28, 16)
```

**Explanation**: 
- The input is 4 images of size 28x28x1 (height, width, channels).
- A 3x3 filter with **same padding** ensures the output height and width remain 28x28.
- The output has 16 feature maps (one per filter), resulting in a shape of (4, 28, 28, 16).

**Visual Description**: 
- A diagram of a 28x28 input image with a 1-pixel zero-padding border, processed by a 3x3 filter to produce 16 feature maps of size 28x28.
- A side-by-side comparison of valid vs. same padding: valid padding yields a 26x26 output, while same padding maintains 28x28.


## Summary Table
| **Padding Type** | **Padding ($ p $)** | **Output Size ($ s = 1 $)** | **Pros** | **Cons** |
|------------------|---------------------|-----------------------------|----------|----------|
| **Valid** | $ p = 0 $ | $ (n - f + 1) \times (n - f + 1) $ | Reduces size, computationally lighter | Loses edge information |
| **Same** | $ p = \lfloor (f - 1)/2 \rfloor $ | $ n \times n $ | Preserves size, retains edge data | Increases computation due to padding |