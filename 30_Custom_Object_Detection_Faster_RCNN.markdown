# Tutorial 30 - Custom Object Detection with Faster R-CNN in CNNs

## 1. What Is Custom Object Detection?
Custom object detection involves training a deep learning model to identify and localize specific objects of interest (e.g., "laptop," "book") in images, using a dataset with custom annotations. Faster R-CNN, a popular object detection model, is well-suited for this task due to its accuracy and flexibility.

- **Process**: Train a Faster R-CNN model on a custom dataset with labeled bounding boxes, then use it to detect and classify objects in new images.
- **Purpose**: Enables applications like inventory tracking (e.g., detecting products), robotics, or personalized image analysis.
- **Why Faster R-CNN?**: Combines a Region Proposal Network (RPN) with a CNN for accurate and efficient detection of custom objects.



## 2. Mechanics of Custom Object Detection with Faster R-CNN
Custom object detection with Faster R-CNN involves preparing a dataset, training the model, and performing inference. The pipeline includes:

1. **Dataset Preparation**:
   - Collect images containing the target objects (e.g., laptops, books).
   - Annotate images with bounding boxes and class labels using tools like LabelImg or VGG Annotator.
   - Format annotations (e.g., in COCO or TFRecord format).
2. **Feature Extraction**: A CNN (e.g., ResNet-50) processes the image to generate feature maps.
3. **Region Proposal Network (RPN)**: Proposes regions likely to contain objects.
4. **Region of Interest (RoI) Pooling**: Extracts fixed-size features from proposed regions.
5. **Classification and Regression**:
   - Classifies each region (e.g., "laptop," "book," or "background").
   - Refines bounding box coordinates (x, y, width, height).
6. **Output**: Bounding boxes with class labels and confidence scores.

**Key Parameters**:
- **Input Size**: Typically 600x600 or larger (resized during preprocessing).
- **Number of Classes**: Number of custom classes (e.g., 2 for "laptop" and "book") plus background.
- **Loss Function**: Combines:
  - RPN loss (for region proposals).
  - Classification loss (cross-entropy for class prediction).
  - Localization loss (smooth L1 for box coordinates).

**Visual Description**: 
- A diagram showing the Faster R-CNN pipeline for custom detection: Input image → CNN (feature maps) → RPN (region proposals) → RoI pooling → Classification and regression → Output with labeled boxes for custom objects.

![RCNN](src\RCNN.webp)

## 3. How Custom Object Detection Works
- **Dataset**: Images with annotations (e.g., XML or JSON files specifying bounding boxes and class labels).
- **Training**:
  - Fine-tune a pre-trained Faster R-CNN model (e.g., trained on COCO) on the custom dataset.
  - Adjust weights for the RPN, classifier, and regressor to learn custom object features.
- **Inference**:
  - Process a new image through the trained model.
  - Output bounding boxes, class labels, and confidence scores for detected objects.
- **Example**: For a dataset of office items, the model learns to detect "laptop" and "book" in various lighting and angles.

**Visual Description**: 
- An animation showing a 600x600 image of a desk with a laptop and book, processed by Faster R-CNN. The model generates feature maps, proposes regions, and outputs bounding boxes labeled "Laptop" and "Book" with confidence scores.



```python
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Paths to model, config, and data
MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
PATH_TO_MODEL = f'{MODEL_NAME}/saved_model'
PATH_TO_LABELS = 'custom_label_map.pbtxt'  # Custom label map (e.g., "laptop," "book")
PATH_TO_IMAGE = 'test_image.jpg'  # Test image with custom objects

# Load pre-trained model
model = tf.saved_model.load(PATH_TO_MODEL)
infer = model.signatures['serving_default']

# Load custom label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=2, use_display_name=True)  # 2 classes: laptop, book
category_index = label_map_util.create_category_index(categories)

# Load and preprocess test image
image = cv2.imread(PATH_TO_IMAGE)
if image is None:
    # Fallback: Dummy image (600x600x3) if test image is unavailable
    image = np.random.randint(0, 255, (600, 600, 3), dtype=np.uint8)
image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.uint8)

# Perform inference
detections = infer(image_tensor)

# Process predictions
boxes = detections['detection_boxes'].numpy()[0]  # (y1, x1, y2, x2)
scores = detections['detection_scores'].numpy()[0]
classes = detections['detection_classes'].numpy()[0].astype(np.int32)

# Print detections with confidence > 0.5
for i in range(len(scores)):
    if scores[i] > 0.5:
        box = boxes[i] * [image.shape[0], image.shape[1], image.shape[0], image.shape[1]]  # Scale to image size
        y1, x1, y2, x2 = box
        label = category_index[classes[i]]['name']
        print(f"Detected: {label} (Confidence: {scores[i]:.2f}) at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# Visualize results
image_with_boxes = image.copy()
vis_util.visualize_boxes_and_labels_on_image_array(
    image_with_boxes,
    boxes,
    classes,
    scores,
    category_index,
    min_score_thresh=0.5,
    use_normalized_coordinates=True
)
cv2.imwrite('output_custom_rcnn.jpg', image_with_boxes)
```

**Output** (example, varies due to image):
```
Detected: laptop (Confidence: 0.92) at (150, 100, 300, 250)
Detected: book (Confidence: 0.88) at (350, 200, 450, 300)
```

**Explanation**:
- **Model**: Pre-trained Faster R-CNN with ResNet-50 backbone (from TensorFlow Model Zoo), fine-tuned for custom classes (e.g., "laptop," "book").
- **Input**: A test image (e.g., `test_image.jpg`) or a dummy 600x600x3 RGB image for demonstration.
- **Label Map**: Defines custom classes (e.g., `custom_label_map.pbtxt` with "laptop" and "book").
- **Output**: Bounding box coordinates, class labels, and confidence scores for detected objects.
- **Visualization**: Draws bounding boxes and labels on the image, saved as `output_custom_rcnn.jpg`.
- **Training Note**: To train on a custom dataset, use the TensorFlow Object Detection API with TFRecords, a configuration file, and a pipeline like:
  - Prepare dataset (images + annotations).
  - Convert to TFRecords.
  - Fine-tune the model using `model_main_tf2.py` (not shown for brevity).

**Visual Description**: 
- A diagram showing a test image (e.g., a desk with a laptop and book) processed by Faster R-CNN, outputting bounding boxes labeled "Laptop" and "Book" with confidence scores.
- A side-by-side comparison of the input image and the output image with drawn bounding boxes.

## 5. Benefits and Challenges
- **Benefits**:
  - **High Accuracy**: Faster R-CNN provides precise detection for custom objects.
  - **Customizability**: Can be trained on any dataset with annotated bounding boxes.
  - **Robustness**: Handles varying object sizes and backgrounds.
- **Challenges**:
  - **Data Preparation**: Requires annotated images (bounding boxes and labels), which can be time-consuming.
  - **Training Complexity**: Fine-tuning needs computational resources and careful configuration.
  - **Speed**: Slower than single-stage detectors like YOLO, less suited for real-time applications.



## 6. Next Steps
- **Dataset Creation**: Use tools like LabelImg to annotate images for custom objects (e.g., create a dataset of office items).
- **Train**: Fine-tune a Faster R-CNN model on your custom dataset using the TensorFlow Object Detection API.
- **Compare**: Test Faster R-CNN vs. other models like YOLO (Tutorial 26) for accuracy and speed.
- **Deploy**: Integrate the model into an application (e.g., a Flask app for web-based detection, similar to your car price prediction project).
- **Visualize**: Plot bounding boxes using OpenCV or Matplotlib to analyze detection results.

**Visual Description**: 
- A screenshot of a Kaggle notebook running the TensorFlow code above, showing the input image and output with bounding boxes for custom objects.
- A plot of detected objects (e.g., laptops and books) from a custom dataset, highlighting bounding boxes and labels.

## 7. Summary Table
| **Aspect** | **Description** | **Pros** | **Cons** |
|------------|-----------------|----------|----------|
| **Purpose** | Detects custom objects with bounding boxes | High accuracy, customizable | Requires annotated dataset |
| **Key Operations** | CNN, RPN, RoI pooling, classification | Robust to object variations | Complex training pipeline |
| **Use Case** | Inventory tracking, robotics, custom vision tasks | Precise localization | Slower than YOLO |
| **Output** | Bounding boxes, custom class labels, confidence scores | Flexible for any object type | Resource-intensive |