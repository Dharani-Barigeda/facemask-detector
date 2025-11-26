import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
from utils import (
    load_face_cascade, detect_faces, preprocess_face, 
    draw_prediction, get_class_labels, load_class_indices,
    ViolationLogger, expand_bbox
)

def detect_masks_in_video(video_source=0, model_path='best_mask_detector.h5',
                          confidence_threshold=0.5, log_violations=True,
                          threshold_mask=None, threshold_no_mask=None, threshold_incorrect=None,
                          incorrect_bias_delta=0.05):
    """
    Detect masks in real-time video stream.
    
    Args:
        video_source: Video source (0 for webcam, or path to video file)
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence for predictions
        log_violations: Whether to log violations
    
    Returns:
        None
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
    
    # Initialize violation logger
    logger = ViolationLogger() if log_violations else None
    
    # Open video source
    print(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")
    
    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting video detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every 30 frames
        if fps_counter == 30:
            fps_end_time = cv2.getTickCount()
            fps = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
            fps_start_time = fps_end_time
            fps_counter = 0
        else:
            fps = 0
        
        # Detect faces
        faces = detect_faces(frame, face_cascade)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Expand bbox to include more lower-face context
            x, y, w, h = expand_bbox(x, y, w, h, frame.shape, pad_ratio=0.15)
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
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
                # Fallback to argmax if none passed thresholds
                predicted_class = int(np.argmax(predictions))
                chosen_label = class_labels.get(predicted_class, 'unknown')
            # Apply bias: if incorrect is close to top prob, prefer incorrect
            top_label = max(probs_by_label, key=probs_by_label.get)
            if 'incorrect_mask' in probs_by_label:
                if probs_by_label['incorrect_mask'] >= (probs_by_label[top_label] - incorrect_bias_delta):
                    chosen_label = 'incorrect_mask'
            confidence = probs_by_label.get(chosen_label, float(np.max(predictions)))

            # Draw and log using the chosen label consistently
            threshold_for_class = default_thresholds.get(chosen_label, confidence_threshold)
            if confidence >= threshold_for_class:
                frame = draw_prediction(frame, (x, y, w, h), chosen_label, confidence)
                if logger and chosen_label in ['no_mask', 'incorrect_mask']:
                    logger.log_violation(chosen_label, confidence, frame, (x, y, w, h))
        
        # Add FPS counter
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Mask Detection - Real Time', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print violation statistics
    if logger:
        stats = logger.get_violation_stats()
        print(f"\nViolation Statistics:")
        print(f"Total violations: {stats.get('total_violations', 0)}")
        print(f"By class: {stats.get('by_class', {})}")

def main():
    parser = argparse.ArgumentParser(description='Detect masks in video stream')
    parser.add_argument('--video', type=str, default='0',
                       help='Video source (0 for webcam, or path to video file)')
    parser.add_argument('--model', type=str, default='best_mask_detector.h5',
                       help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--no_log', action='store_true',
                       help='Do not log violations')
    parser.add_argument('--th_mask', type=float, default=None,
                       help='Threshold for class: mask')
    parser.add_argument('--th_no_mask', type=float, default=None,
                       help='Threshold for class: no_mask')
    parser.add_argument('--th_incorrect', type=float, default=None,
                       help='Threshold for class: incorrect_mask')
    parser.add_argument('--incorrect_bias_delta', type=float, default=0.05,
                       help='Prefer incorrect_mask if within delta of top probability')
    
    args = parser.parse_args()
    
    # Convert video source to int if it's a number
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first using train_mask_detector.py")
        return
    
    try:
        # Start video detection
        detect_masks_in_video(
            video_source,
            args.model,
            args.confidence,
            not args.no_log,
            args.th_mask,
            args.th_no_mask,
            args.th_incorrect,
            args.incorrect_bias_delta
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

