# Tutorial 31 - One-Shot Learning for Object Detection in CNNs

## 1. What Is One-Shot Learning?
One-shot learning is a machine learning approach that enables a model to recognize or detect objects after seeing only one (or a few) examples, unlike traditional methods requiring large datasets. In object detection, one-shot learning aims to identify and localize objects (e.g., "laptop," "book") using minimal training data.

- **Process**: The model learns to compare a new image (query) with a single example (support image) to detect and classify objects, often using similarity metrics or learned embeddings.
- **Purpose**: Useful in scenarios with limited data, such as detecting rare objects or personalized items (e.g., a specific toy or tool).
- **Key Idea**: Instead of learning specific classes, the model learns to measure similarity between images, enabling generalization from one example.



## 2. Mechanics of One-Shot Learning for Object Detection
One-shot learning for object detection typically adapts CNN-based models (e.g., Faster R-CNN) or uses specialized architectures like Siamese Networks or Relation Networks. The process includes:

1. **Support Set**: A small set of labeled examples (e.g., one image per class with bounding box annotations).
2. **Query Image**: The input image where objects need to be detected.
3. **Feature Extraction**: A CNN extracts features from both support and query images.
4. **Similarity Comparison**:
   - Compare feature embeddings of support and query regions to identify matching objects.
   - Use metrics like cosine similarity or learned relation modules.
5. **Bounding Box Prediction**: Localize objects in the query image by adapting region proposals (e.g., using an RPN from Faster R-CNN).
6. **Output**: Bounding boxes with class labels and confidence scores based on similarity to the support set.

**Common Approaches**:
- **Siamese Networks**: Use twin CNNs to compare support and query images, predicting similarity for detection.
- **Prototypical Networks**: Compute a prototype embedding for each class from the support set and compare it to query regions.
- **Faster R-CNN with Few-Shot Adaptation**: Fine-tune a pre-trained Faster R-CNN model to perform detection with one example per class.

**Key Parameters**:
- **Support Set Size**: Typically 1 (one-shot) or a few examples per class.
- **Input Size**: Varies (e.g., 600x600x3 for RGB images).
- **Loss Function**: Combines similarity loss (e.g., contrastive loss) and localization loss (e.g., smooth L1 for bounding boxes).

**Visual Description**: 
- A diagram showing the one-shot learning pipeline: Support image (e.g., laptop with bounding box) → CNN feature extraction → Comparison with query image features → Bounding box and label prediction in the query image.

![One shot learning](src\One_shot_learning.avif)

## 3. How One-Shot Learning Works
- **Support Set**: A single image per class with annotated bounding boxes (e.g., one laptop image).
- **Query Image**: A new image where objects need to be detected (e.g., a desk with multiple objects).
- **Training**:
  - Fine-tune a pre-trained CNN (e.g., Faster R-CNN) to learn similarity between support and query images.
  - Use meta-learning or few-shot learning techniques to generalize from one example.
- **Inference**:
  - Extract features from the support image and query image using a CNN.
  - Compare features to identify regions in the query image that match the support image’s object.
  - Predict bounding boxes and confidence scores for matching regions.
- **Example**: Given one image of a laptop, the model detects laptops in a new image by comparing features, even if the laptop’s appearance varies slightly.

**Visual Description**: 
- An animation showing a support image (e.g., a laptop) and a query image (e.g., a desk scene). The model extracts features, compares them, and outputs a bounding box labeled "Laptop" in the query image.

## 4. Code Example
Below is a simplified TensorFlow code example demonstrating one-shot learning for object detection using a pre-trained Faster R-CNN model adapted for a custom class (e.g., "laptop") with a single support image. This example uses the TensorFlow Object Detection API and assumes a minimal setup for demonstration purposes.

```python
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Paths to model, label map, support, and query images
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
PATH_TO_MODEL = f'{MODEL_NAME}/saved_model'
PATH_TO_LABELS = 'custom_label_map.pbtxt'  # Single class: "laptop"
PATH_TO_SUPPORT = 'support_laptop.jpg'  # Support image with one laptop
PATH_TO_QUERY = 'query_image.jpg'  # Query image with potential laptops

# Load pre-trained Faster R-CNN model
model = tf.saved_model.load(PATH_TO_MODEL)
infer = model.signatures['serving_default']

# Load custom label map (single class for one-shot learning)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=1, use_display_name=True)  # 1 class: laptop
category_index = label_map_util.create_category_index(categories)

# Load and preprocess support and query images
support_image = cv2.imread(PATH_TO_SUPPORT)
query_image = cv2.imread(PATH_TO_QUERY)
if support_image is None or query_image is None:
    # Fallback: Dummy images (600x600x3)
    support_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
    query_image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)

support_tensor = tf.convert_to_tensor(support_image[None, ...], dtype=tf.uint8)
query_tensor = tf.convert_to_tensor(query_image[None, ...], dtype=tf.uint8)

# Perform inference on query image
detections = infer(query_tensor)

# Process predictions (simplified one-shot comparison)
boxes = detections['detection_boxes'].numpy()[0]  # (y1, x1, y2, x2)
scores brotherhoods = detections['detection_scores'].numpy()[0]
classes = detections['detection_classes'].numpy()[0].astype(np.int32)

# Filter detections for the custom class (e.g., "laptop") with confidence > 0.5
for i in range(len(scores)):
    if scores[i] > 0.5 and classes[i] == 1:  # Assuming class ID 1 is "laptop"
        box = boxes[i] * [query_image.shape[0], query_image.shape[1], 
                          query_image.shape[0], query_image.shape[1]]  # Scale to image size
        y1, x1, y2, x2 = box
        label = category_index[classes[i]]['name']
        print(f"Detected: {label} (Confidence: {scores[i]:.2f}) at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# Visualize results on query image
image_with_boxes = query_image.copy()
vis_util.visualize_boxes_and_labels_on_image_array(
    image_with_boxes,
    boxes,
    classes,
    scores,
    category_index,
    min_score_thresh=0.5,
    use_normalized_coordinates=True
)
cv2.imwrite('output_one_shot.jpg', image_with_boxes)
```

**Output** (example, varies due to image):
```
Detected: laptop (Confidence: 0.90) at (150, 100, 300, 250)
```

**Explanation**:
- **Model**: Pre-trained Faster R-CNN with ResNet-50 backbone, fine-tuned (conceptually) for one-shot learning on a custom class ("laptop").
- **Support Image**: A single image with a laptop and bounding box annotation (e.g., `support_laptop.jpg`).
- **Query Image**: A test image (e.g., `query_image.jpg`) where laptops need to be detected.
- **Inference**: The model detects objects in the query image, filtering for the custom class based on similarity to the support image (simplified here; real one-shot learning requires additional similarity metrics or meta-learning).
- **Visualization**: Draws bounding boxes and labels on the query image, saved as `output_one_shot.jpg`.
- **Training Note**: For true one-shot learning, fine-tune the model using a meta-learning framework (e.g., Model-Agnostic Meta-Learning) or a Siamese Network on a dataset with one example per class. This example assumes a pre-trained model for simplicity.



## 5. Benefits and Challenges
- **Benefits**:
  - **Minimal Data**: Detects objects with only one example per class, ideal for rare or custom objects.
  - **Flexibility**: Applicable to personalized tasks (e.g., detecting specific tools or items).
  - **Accuracy**: Leverages powerful CNNs like Faster R-CNN for precise localization.
- **Challenges**:
  - **Training Complexity**: Requires meta-learning or similarity-based architectures, which are harder to train than standard object detection.
  - **Generalization**: May struggle with significant variations in object appearance (e.g., different laptop models).
  - **Annotation**: Still needs at least one annotated support image per class.


## 6. Next Steps
- **Dataset Creation**: Create a small dataset with one or a few images per custom class using annotation tools like LabelImg.
- **Train**: Implement a one-shot learning framework (e.g., Siamese Network or meta-learning) to fine-tune Faster R-CNN on your dataset.
- **Compare**: Test one-shot learning vs. traditional object detection (Tutorial 27) for accuracy and efficiency.
- **Deploy**: Integrate the model into an application (e.g., a Flask app for real-time detection, similar to your car price prediction project).
- **Visualize**: Use OpenCV or Matplotlib to visualize bounding boxes and analyze detection results.



## 7. Summary Table
| **Aspect** | **Description** | **Pros** | **Cons** |
|------------|-----------------|----------|----------|
| **Purpose** | Detects objects with one example per class | Minimal data needed, highly flexible | Limited generalization to variations |
| **Key Operations** | Feature extraction, similarity comparison, box prediction | Accurate with few examples | Complex training setup |
| **Use Case** | Rare objects, personalized detection | Ideal for custom tasks | Requires meta-learning expertise |
| **Output** | Bounding boxes, custom class labels, confidence scores | Fast adaptation to new classes | May miss dissimilar objects |