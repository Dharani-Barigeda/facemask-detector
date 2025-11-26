import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json

def load_face_cascade():
    """
    Load OpenCV Haar cascade for face detection.
    
    Returns:
        cv2.CascadeClassifier object
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise ValueError("Could not load face cascade classifier")
    
    return face_cascade

def detect_faces(image, face_cascade, min_face_size=(30, 30)):
    """
    Detect faces in an image.
    
    Args:
        image: Input image (BGR format)
        face_cascade: OpenCV face cascade classifier
        min_face_size: Minimum face size for detection
    
    Returns:
        List of face rectangles (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def expand_bbox(x, y, w, h, image_shape, pad_ratio=0.15):
    """
    Expand a bounding box by a ratio to include more context.
    Ensures the box stays within image bounds.
    """
    ih, iw = image_shape[:2]
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)
    x_new = max(0, x - pad_w)
    y_new = max(0, y - pad_h)
    w_new = min(iw - x_new, w + 2 * pad_w)
    h_new = min(ih - y_new, h + 2 * pad_h)
    return x_new, y_new, w_new, h_new

def preprocess_face(face_roi, target_size=(224, 224)):
    """
    Preprocess face region for model input.
    
    Args:
        face_roi: Face region of interest
        target_size: Target size for the face image
    
    Returns:
        Preprocessed face image
    """
    # Resize face to target size
    face_resized = cv2.resize(face_roi, target_size)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    face_normalized = face_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def get_class_labels():
    """
    Get class labels for mask detection.
    
    Returns:
        Dictionary mapping class indices to labels
    """
    return {
        0: 'mask',
        1: 'no_mask', 
        2: 'incorrect_mask'
    }

def get_class_colors():
    """
    Get colors for each class for visualization.
    
    Returns:
        Dictionary mapping class labels to BGR colors
    """
    return {
        'mask': (0, 255, 0),        # Green
        'no_mask': (0, 0, 255),     # Red
        'incorrect_mask': (0, 165, 255)  # Orange
    }

def draw_prediction(image, face_rect, prediction, confidence):
    """
    Draw prediction on the image.
    
    Args:
        image: Input image
        face_rect: Face rectangle (x, y, w, h)
        prediction: Predicted class
        confidence: Prediction confidence
    
    Returns:
        Image with prediction drawn
    """
    x, y, w, h = face_rect
    colors = get_class_colors()
    color = colors.get(prediction, (255, 255, 255))
    
    # Draw rectangle around face
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Prepare label text
    label = f"{prediction}: {confidence:.2f}"
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x, y - text_height - 10),
        (x + text_width, y),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        label,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return image

class ViolationLogger:
    """
    Class for logging mask violations.
    """
    
    def __init__(self, log_file='violations.csv', snapshot_dir='violation_snapshots'):
        self.log_file = log_file
        self.snapshot_dir = snapshot_dir
        
        # Create snapshot directory if it doesn't exist
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(log_file):
            df = pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'image_path'])
            df.to_csv(log_file, index=False)
    
    def log_violation(self, prediction, confidence, image, face_rect):
        """
        Log a mask violation.
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            image: Image containing the violation
            face_rect: Face rectangle coordinates
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save snapshot
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        snapshot_filename = f"violation_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, face_roi)
        
        # Log to CSV
        new_row = {
            'timestamp': timestamp,
            'prediction': prediction,
            'confidence': confidence,
            'image_path': snapshot_path
        }
        
        df = pd.read_csv(self.log_file)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.log_file, index=False)
        
        print(f"Violation logged: {prediction} at {timestamp}")
    
    def get_violation_stats(self):
        """
        Get violation statistics.
        
        Returns:
            Dictionary with violation statistics
        """
        if not os.path.exists(self.log_file):
            return {}
        
        df = pd.read_csv(self.log_file)
        
        stats = {
            'total_violations': len(df),
            'by_class': df['prediction'].value_counts().to_dict(),
            'recent_violations': df.tail(10).to_dict('records')
        }
        
        return stats

def load_class_indices(file_path='class_indices.json'):
    """
    Load class indices from JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Dictionary mapping class names to indices
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using default class indices.")
        return {'mask': 0, 'no_mask': 1, 'incorrect_mask': 2}
    
    with open(file_path, 'r') as f:
        return json.load(f)

def create_dataset_structure(base_dir='dataset'):
    """
    Create the expected dataset directory structure.
    
    Args:
        base_dir: Base directory for the dataset
    """
    subdirs = ['train', 'val']
    classes = ['mask', 'no_mask', 'incorrect_mask']
    
    for subdir in subdirs:
        for class_name in classes:
            dir_path = os.path.join(base_dir, subdir, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    print(f"Dataset structure created in: {base_dir}")
    print("Please add your images to the appropriate subdirectories.")

