# Notes on Tutorial 19 - Training Artificial Neural Network using Google Colab GPU

## 1. Why Use Google Colab for ANN Training?
Google Colab provides free cloud-based GPUs/TPUs, ideal for training deep learning models with high computational needs.

- **Benefits**:
  - Free access to NVIDIA GPUs (e.g., K80, T4).
  - Pre-installed libraries (TensorFlow, Keras, PyTorch).
  - Scalable for large datasets/models.
- **Setup**: Enable GPU in Colab (Runtime → Change runtime type → GPU).

## 2. Steps to Train ANN in Colab
1. **Set Up Environment**:
   - Create a new Colab notebook.
   - Enable GPU runtime.
   - Install dependencies (usually pre-installed).
2. **Load Data**: Use NumPy, Pandas, or TensorFlow datasets.
3. **Build Model**: Define ANN using Keras.
4. **Train**: Use GPU for faster epochs.
5. **Evaluate/Save**: Check accuracy, save model weights.

**Visual Description**: Flowchart: create notebook → enable GPU → load data → train → save model. Colab notebook screenshot.


## 3. Tips for Colab Training
- **Check GPU**: Use `tf.config.list_physical_devices('GPU')`.
- **Batch Size**: Use 32–128 for GPU efficiency.
- **Save Models**: Use `.h5` format or Google Drive integration.
- **Avoid Timeouts**: Run small cells, use Google Drive for data.
