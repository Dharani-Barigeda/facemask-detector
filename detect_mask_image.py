import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
from utils import (
    load_face_cascade, detect_faces, preprocess_face, 
    draw_prediction, get_class_labels, load_class_indices,
    expand_bbox
)

def detect_masks_in_image(image_path, model_path='best_mask_detector.h5', 
                         confidence_threshold=0.5, save_output=True,
                         threshold_mask=None, threshold_no_mask=None, threshold_incorrect=None,
                         incorrect_bias_delta=0.05):
    """
    Detect masks in a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence for predictions
        save_output: Whether to save the output image
    
    Returns:
        Image with predictions drawn
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load class indices
    class_indices = load_class_indices()
    class_labels = {v: k for k, v in class_indices.items()}

    # Class-specific thresholds (fallback to global)
    default_thresholds = {
        'mask': confidence_threshold,
        'no_mask': confidence_threshold,
        'incorrect_mask': confidence_threshold
    }
    if threshold_mask is not None:
        default_thresholds['mask'] = threshold_mask
    if threshold_no_mask is not None:
        default_thresholds['no_mask'] = threshold_no_mask
    if threshold_incorrect is not None:
        default_thresholds['incorrect_mask'] = threshold_incorrect
    
    # Load face cascade
    face_cascade = load_face_cascade()
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Detect faces
    faces = detect_faces(image, face_cascade)
    print(f"Found {len(faces)} face(s)")
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Expand bbox to include more lower-face context
        x, y, w, h = expand_bbox(x, y, w, h, image.shape, pad_ratio=0.15)
        # Extract face region
        face_roi = image[y:y+h, x:x+w]
        
        # Preprocess face
        face_processed = preprocess_face(face_roi)
        
        # Make prediction
        predictions = model.predict(face_processed, verbose=0)[0]
        probs_by_index = {idx: float(prob) for idx, prob in enumerate(predictions)}
        probs_by_label = {class_labels[i]: p for i, p in probs_by_index.items() if i in class_labels}

        # Decide final label using per-class thresholds with priority: incorrect_mask > no_mask > mask
        chosen_label = None
        for candidate in ['incorrect_mask', 'no_mask', 'mask']:
            if candidate in probs_by_label:
                if probs_by_label[candidate] >= default_thresholds.get(candidate, confidence_threshold):
                    chosen_label = candidate
                    break
        if chosen_label is None:
            predicted_class = int(np.argmax(predictions))
            chosen_label = class_labels.get(predicted_class, 'unknown')
        # Apply bias: if incorrect is close to top prob, prefer incorrect
        top_label = max(probs_by_label, key=probs_by_label.get)
        if 'incorrect_mask' in probs_by_label:
            if probs_by_label['incorrect_mask'] >= (probs_by_label[top_label] - incorrect_bias_delta):
                chosen_label = 'incorrect_mask'
        confidence = probs_by_label.get(chosen_label, float(np.max(predictions)))

        print(f"Face {i+1}: {chosen_label} (confidence: {confidence:.3f})")

        # Draw prediction if confidence is above class-specific threshold
        threshold_for_class = default_thresholds.get(chosen_label, confidence_threshold)
        if confidence >= threshold_for_class:
            image = draw_prediction(image, (x, y, w, h), chosen_label, confidence)
        else:
            print(f"  Low confidence prediction ignored (threshold: {confidence_threshold})")
    
    # Save output image
    if save_output:
        output_path = f"output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"Output saved as: {output_path}")
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Detect masks in an image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='best_mask_detector.h5',
                       help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save output image')
    parser.add_argument('--th_mask', type=float, default=None,
                       help='Threshold for class: mask')
    parser.add_argument('--th_no_mask', type=float, default=None,
                       help='Threshold for class: no_mask')
    parser.add_argument('--th_incorrect', type=float, default=None,
                       help='Threshold for class: incorrect_mask')
    parser.add_argument('--incorrect_bias_delta', type=float, default=0.05,
                       help='Prefer incorrect_mask if within delta of top probability')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using train_mask_detector.py")
        return
    
    try:
        # Detect masks
        result_image = detect_masks_in_image(
            args.image, 
            args.model, 
            args.confidence,
            not args.no_save,
            args.th_mask,
            args.th_no_mask,
            args.th_incorrect,
            args.incorrect_bias_delta
        )
        
        # Display result
        cv2.imshow('Face Mask Detection', result_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

