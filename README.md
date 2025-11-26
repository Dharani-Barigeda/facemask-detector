# FaceMask-Plus: Advanced Face Mask Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.5+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A production-ready, deep learning-based face mask detection system that accurately classifies mask usage into three categories: **proper mask**, **no mask**, and **incorrect mask** (e.g., mask below nose/mouth). Built with TensorFlow, OpenCV, and Streamlit for real-time detection and web-based deployment.

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage](#-usage)
- [Advanced Configuration](#-advanced-configuration)
- [Results](#-results)
- [Performance Optimization](#-performance-optimization)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

FaceMask-Plus is an intelligent computer vision system designed to detect and classify face mask usage in real-time. The system uses a fine-tuned MobileNetV2 architecture to achieve high accuracy in distinguishing between proper mask usage, no mask, and incorrect mask placement (such as masks worn below the nose or mouth).

### Use Cases

- **Public Health Monitoring**: Automated mask compliance checking in public spaces
- **Security Systems**: Integration into surveillance systems for safety protocols
- **Access Control**: Entry point monitoring for mask requirements
- **Research & Analytics**: Data collection on mask usage patterns

---

## 🎯 Key Features

### Core Capabilities

- **3-Class Classification**: Accurately distinguishes between:
  - ✅ **Mask**: Properly worn mask covering nose and mouth
  - ❌ **No Mask**: Face without any mask
  - ⚠️ **Incorrect Mask**: Mask present but worn incorrectly (below nose/mouth, on chin, etc.)

- **Real-Time Detection**: 
  - Live webcam feed processing
  - Video file support
  - High FPS performance with optimized inference

- **Advanced Detection Features**:
  - **Padded ROI Extraction**: Enhanced face region extraction with 15% padding to capture more context around the face, improving incorrect mask detection
  - **Class-Specific Thresholds**: Fine-tune sensitivity for each class independently
  - **Incorrect Mask Bias**: Intelligent decision logic that prioritizes incorrect_mask detection when probabilities are close
  - **Class Weighting**: Automatic class imbalance handling during training

### User Interfaces

- **Command-Line Tools**: 
  - Image detection script
  - Real-time video detection with violation logging
  - Model training and conversion utilities

- **Web Interface**: 
  - Streamlit-based interactive demo
  - Real-time image upload and processing
  - Advanced threshold controls
  - Debug mode with per-class probability visualization

### Production Features

- **Violation Logging**: Automatic CSV logging of mask violations with timestamps
- **Snapshot Capture**: Saves violation images for audit trails
- **Model Export**: TensorFlow Lite conversion for mobile/edge deployment
- **Docker Support**: Containerized deployment ready
- **Training Visualization**: Automatic generation of training history plots and confusion matrices

---

## 🏗️ Architecture

### Model Architecture

The system uses a **transfer learning** approach with MobileNetV2 as the base architecture:

```
Input (224×224×3)
    ↓
MobileNetV2 (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(128, ReLU) + Dropout(0.5)
    ↓
Dense(3, Softmax) → [mask, no_mask, incorrect_mask]
```

**Training Strategy**:
1. **Phase 1**: Freeze MobileNetV2 base, train only classification head
2. **Phase 2**: Unfreeze base model, fine-tune with lower learning rate
3. **Class Weighting**: Automatic balancing of imbalanced datasets

### Detection Pipeline

```
Image/Video Frame
    ↓
Face Detection (Haar Cascade)
    ↓
Face ROI Extraction (with 15% padding)
    ↓
Preprocessing (Resize → RGB → Normalize)
    ↓
Model Inference
    ↓
Class-Specific Threshold Decision
    ↓
Visualization & Logging
```

### Key Technical Components

- **Face Detection**: OpenCV Haar Cascade Classifier
- **Deep Learning Framework**: TensorFlow/Keras
- **Base Model**: MobileNetV2 (ImageNet pre-trained)
- **Image Processing**: OpenCV, NumPy, PIL
- **Web Framework**: Streamlit

---

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (for real-time detection)
- 4GB+ RAM recommended
- GPU optional but recommended for faster training

### Step-by-Step Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Dharani-Barigeda/facemask-detector.git
cd facemask-detector
```

2. **Create a virtual environment** (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import tensorflow as tf; import cv2; import streamlit; print('All dependencies installed successfully!')"
```

### Dependencies

- `tensorflow>=2.5` - Deep learning framework
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `streamlit` - Web interface
- `Pillow` - Image processing
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualization
- `tqdm` - Progress bars
- `imutils` - Image utilities

---

## 📊 Dataset Preparation

### Directory Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── mask/              # Images with properly worn masks
│   ├── no_mask/           # Images without masks
│   └── incorrect_mask/    # Images with incorrectly worn masks
└── val/
    ├── mask/
    ├── no_mask/
    └── incorrect_mask/
```

### Dataset Requirements

**Minimum Recommended**:
- 100 images per class per split (train/val)
- Total: ~600 images minimum

**Optimal Performance**:
- 500+ images per class per split
- Total: 3000+ images
- Balanced distribution across classes

**Image Guidelines**:
- Diverse lighting conditions
- Various angles and poses
- Different mask types and colors
- Multiple ethnicities and age groups
- Clear visibility of face and mask area

### Creating Dataset Structure

You can use the setup script to create the directory structure:

```bash
python setup.py
```

Or manually create directories:

```bash
mkdir -p dataset/train/{mask,no_mask,incorrect_mask}
mkdir -p dataset/val/{mask,no_mask,incorrect_mask}
```

---

## 🚀 Usage

### 1. Training the Model

**Basic Training**:
```bash
python train_mask_detector.py --data_dir dataset --epochs 50
```

**Advanced Training Options**:
```bash
# Custom batch size and image size
python train_mask_detector.py --data_dir dataset --epochs 50 --batch_size 32 --img_size 224

# Quick training run (for testing)
python train_mask_detector.py --data_dir dataset --epochs 20
```

**Training Outputs**:
- `best_mask_detector.h5` - Best model weights (saved based on validation accuracy)
- `class_indices.json` - Class label mappings
- `training_history.png` - Training/validation accuracy and loss plots
- `confusion_matrix.png` - Classification performance matrix

**Training Features**:
- Automatic class weight computation for imbalanced datasets
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Model checkpointing (saves best model automatically)

### 2. Real-Time Video Detection

**Webcam Detection**:
```bash
python detect_mask_video.py
```

**Custom Video Source**:
```bash
# Use different camera index
python detect_mask_video.py --video 1

# Process video file
python detect_mask_video.py --video path/to/video.mp4
```

**Advanced Options**:
```bash
# Custom confidence thresholds per class
python detect_mask_video.py \
    --confidence 0.5 \
    --th_mask 0.6 \
    --th_no_mask 0.5 \
    --th_incorrect 0.35 \
    --incorrect_bias_delta 0.08

# Disable violation logging
python detect_mask_video.py --no_log
```

**Controls**:
- Press `q` to quit
- Press `s` to save current frame

**Outputs**:
- Real-time video display with bounding boxes and labels
- `violations.csv` - Log of all violations with timestamps
- `violation_snapshots/` - Directory containing violation images

### 3. Image Detection

**Basic Usage**:
```bash
python detect_mask_image.py --image path/to/image.jpg
```

**Advanced Options**:
```bash
# Custom thresholds
python detect_mask_image.py \
    --image path/to/image.jpg \
    --confidence 0.5 \
    --th_incorrect 0.35 \
    --incorrect_bias_delta 0.08

# Don't save output image
python detect_mask_image.py --image path/to/image.jpg --no_save
```

**Outputs**:
- Annotated image displayed in window
- `output_<original_filename>.jpg` - Saved result image

### 4. Web Interface (Streamlit)

**Launch the App**:
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

**Features**:
- Upload and process images
- Adjustable confidence thresholds
- Advanced per-class threshold controls
- Debug mode with probability visualization
- Real-time detection results

**Usage in Streamlit**:
1. Select model file (default: `best_mask_detector.h5`)
2. Adjust global confidence threshold
3. (Optional) Expand "Advanced Thresholds" for fine-tuning
4. Upload an image
5. View detection results with bounding boxes and labels

### 5. Model Conversion to TensorFlow Lite

**Convert to TFLite** (with quantization):
```bash
python convert_to_tflite.py --model best_mask_detector.h5
```

**Convert without Quantization**:
```bash
python convert_to_tflite.py --model best_mask_detector.h5 --no_quantize
```

**Test Converted Model**:
```bash
python convert_to_tflite.py --model best_mask_detector.h5 --test
```

**Output**: `best_mask_detector.tflite` - Optimized model for mobile/edge devices

---

## ⚙️ Advanced Configuration

### Class-Specific Thresholds

Fine-tune detection sensitivity for each class:

- **`--th_mask`**: Threshold for proper mask detection (default: same as `--confidence`)
- **`--th_no_mask`**: Threshold for no mask detection (default: same as `--confidence`)
- **`--th_incorrect`**: Threshold for incorrect mask detection (recommended: 0.30-0.40)

**Example**:
```bash
python detect_mask_video.py \
    --th_mask 0.65 \
    --th_no_mask 0.5 \
    --th_incorrect 0.35
```

### Incorrect Mask Bias Delta

The `--incorrect_bias_delta` parameter helps prioritize incorrect_mask detection when probabilities are close:

```bash
python detect_mask_video.py --incorrect_bias_delta 0.08
```

This means: if incorrect_mask probability is within 0.08 of the top probability, prefer incorrect_mask.

### Training Hyperparameters

**Batch Size**: Adjust based on available memory
- Smaller batch (16): More stable gradients, slower training
- Larger batch (32-64): Faster training, requires more memory

**Image Size**: Default 224×224 (MobileNetV2 standard)
- Smaller (160×160): Faster inference, lower accuracy
- Larger (320×320): Better accuracy, slower inference

**Epochs**: 
- Quick test: 10-20 epochs
- Production: 50-100 epochs (with early stopping)

---

## 📈 Results

### Model Performance

*[Add your training results, accuracy metrics, confusion matrix images, and sample detection results here]*

### Sample Detections

*[Add screenshots or images showing:*
- *Proper mask detection (green box)*
- *No mask detection (red box)*
- *Incorrect mask detection (orange box)*
- *Streamlit interface*
- *Training curves*
- *Confusion matrix*]

---

## 🚀 Performance Optimization

### For Better Accuracy

1. **Increase Training Data**:
   - Collect more diverse images
   - Ensure balanced class distribution
   - Include edge cases (various mask types, lighting conditions)

2. **Training Improvements**:
   - Increase epochs (with early stopping)
   - Fine-tune hyperparameters
   - Use data augmentation (already included)

3. **Threshold Tuning**:
   - Lower incorrect_mask threshold (0.30-0.40)
   - Adjust based on your use case requirements

### For Faster Inference

1. **Use TFLite Model**:
   - Quantized model is smaller and faster
   - Better for mobile/edge deployment

2. **Reduce Input Size**:
   - Use smaller image size (160×160) if accuracy allows

3. **GPU Acceleration**:
   - Install TensorFlow with GPU support
   - Significantly faster training and inference

### For Production Deployment

1. **Error Handling**: Already implemented in scripts
2. **Model Versioning**: Save model with version numbers
3. **Monitoring**: Use violation logs for analytics
4. **Scalability**: Consider using Docker containers

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t facemask-plus .
```

### Run Container

```bash
# Basic run
docker run -p 8501:8501 facemask-plus

# With volume mounting for model persistence
docker run -p 8501:8501 -v $(pwd)/models:/app/models facemask-plus

# With custom port
docker run -p 8080:8501 facemask-plus
```

### Docker Compose (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  facemask-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./violation_snapshots:/app/violation_snapshots
```

Run with:
```bash
docker-compose up
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Model file not found"
**Problem**: `best_mask_detector.h5` doesn't exist

**Solution**: 
- Train the model first: `python train_mask_detector.py --data_dir dataset --epochs 20`
- Check if file exists: `ls best_mask_detector.h5` (Linux/Mac) or `dir best_mask_detector.h5` (Windows)

#### 2. "No faces detected"
**Problem**: Face detection fails

**Solutions**:
- Ensure good lighting conditions
- Check if face is clearly visible and not occluded
- Try adjusting camera angle
- Increase face size in frame
- Check camera permissions

#### 3. Low Accuracy / Incorrect Predictions
**Problem**: Model predictions are inaccurate

**Solutions**:
- Retrain with more diverse data
- Check data quality and labeling
- Adjust confidence thresholds
- Lower incorrect_mask threshold: `--th_incorrect 0.35`
- Use incorrect_bias_delta: `--incorrect_bias_delta 0.08`

#### 4. Webcam Not Working
**Problem**: Cannot access webcam

**Solutions**:
- Check camera permissions in system settings
- Try different video source index: `--video 1` or `--video 2`
- Close other applications using the camera
- On Linux, check video device: `ls /dev/video*`

#### 5. TensorFlow Installation Issues
**Problem**: TensorFlow installation fails

**Solutions**:
- Ensure Python 3.7-3.10 (TensorFlow 2.x compatibility)
- Use 64-bit Python
- Install CPU version: `pip install tensorflow`
- For GPU: `pip install tensorflow-gpu` (requires CUDA)

#### 6. Streamlit Not Opening
**Problem**: Browser doesn't open automatically

**Solution**: 
- Check terminal for URL (usually `http://localhost:8501`)
- Manually open the URL in browser
- Check firewall settings

#### 7. Out of Memory Errors
**Problem**: Training crashes due to memory

**Solutions**:
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--img_size 160`
- Close other applications
- Use smaller dataset subset for testing

#### 8. Incorrect Mask Not Detected
**Problem**: Orange box doesn't appear

**Solutions**:
- Lower incorrect_mask threshold: `--th_incorrect 0.30`
- Use bias delta: `--incorrect_bias_delta 0.10`
- Check debug probabilities in Streamlit
- Ensure training data has diverse incorrect_mask examples
- Retrain model with class weighting (already implemented)

---

## 📁 Project Structure

```
facemask-detector/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup script
├── Dockerfile                  # Docker configuration
│
├── mask_model.py               # Model architecture definition
├── train_mask_detector.py      # Training script
├── utils.py                    # Utility functions
│
├── detect_mask_image.py        # Image detection script
├── detect_mask_video.py        # Real-time video detection
├── streamlit_app.py            # Web interface
├── convert_to_tflite.py        # Model conversion utility
│
├── dataset/                    # Training dataset (not in repo)
│   ├── train/
│   │   ├── mask/
│   │   ├── no_mask/
│   │   └── incorrect_mask/
│   └── val/
│       ├── mask/
│       ├── no_mask/
│       └── incorrect_mask/
│
├── best_mask_detector.h5        # Trained model (generated)
├── class_indices.json          # Class mappings (generated)
├── training_history.png         # Training plots (generated)
├── confusion_matrix.png         # Confusion matrix (generated)
│
├── violations.csv              # Violation logs (generated)
└── violation_snapshots/        # Violation images (generated)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update README if adding new features
- Write clear commit messages
- Test your changes thoroughly

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Dharani Barigeda**

- GitHub: [@Dharani-Barigeda](https://github.com/Dharani-Barigeda)
- Repository: [facemask-detector](https://github.com/Dharani-Barigeda/facemask-detector)

---

## 🙏 Acknowledgments

- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library
- **Streamlit** - Web application framework
- **MobileNetV2** - Efficient base architecture
- Open source community for inspiration and support

---

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on [GitHub](https://github.com/Dharani-Barigeda/facemask-detector/issues)
- Check existing issues for solutions

---

<div align="center">

**FaceMask-Plus** - Built with ❤️ using TensorFlow, OpenCV, and Streamlit

⭐ Star this repo if you find it helpful!

</div>
