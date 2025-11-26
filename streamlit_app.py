import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
from utils import (
    load_face_cascade, detect_faces, preprocess_face, 
    draw_prediction, get_class_labels, load_class_indices,
    expand_bbox
)

# Page config
st.set_page_config(
    page_title="FaceMask-Plus Detection",
    page_icon="😷",
    layout="wide"
)

@st.cache(allow_output_mutation=True)
def load_model_cached(model_path):
    """Load model with caching."""
    return load_model(model_path)

def detect_masks_in_image(image, model, confidence_threshold=0.5,
                          threshold_mask=None, threshold_no_mask=None, threshold_incorrect=None,
                          return_probs=False, incorrect_bias_delta=0.05):
    """
    Detect masks in an image using the loaded model.
    
    Args:
        image: Input image (numpy array)
        model: Loaded Keras model
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        Tuple of (processed_image, predictions_info)
    """
    # Load class indices
    class_indices = load_class_indices()
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Load face cascade
    face_cascade = load_face_cascade()
    
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect faces
    faces = detect_faces(image, face_cascade)
    
    predictions_info = []
    processed_image = image.copy()
    
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

        # Store prediction info
        info = {
            'face_id': i + 1,
            'class': chosen_label,
            'confidence': confidence,
            'bbox': (x, y, w, h)
        }
        if return_probs:
            info['probs'] = probs_by_label
        predictions_info.append(info)

        # Draw prediction if confidence is above class-specific threshold
        threshold_for_class = default_thresholds.get(chosen_label, confidence_threshold)
        if confidence >= threshold_for_class:
            processed_image = draw_prediction(processed_image, (x, y, w, h), chosen_label, confidence)
    
    return processed_image, predictions_info

def main():
    st.title("😷 FaceMask-Plus Detection")
    st.markdown("Upload an image to detect face masks using our AI model.")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["best_mask_detector.h5"],
        help="Choose the trained model to use for detection"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence required for predictions"
    )

    with st.sidebar.expander("Advanced Thresholds", expanded=False):
        st.caption("Override per-class thresholds to make a class easier/harder to trigger.")
        th_mask = st.slider("Threshold: mask", 0.1, 1.0, confidence_threshold, 0.05)
        th_no_mask = st.slider("Threshold: no_mask", 0.1, 1.0, confidence_threshold, 0.05)
        th_incorrect = st.slider("Threshold: incorrect_mask", 0.1, 1.0, max(0.3, confidence_threshold - 0.1), 0.05)
        show_debug = st.checkbox("Show per-class probabilities (debug)", value=False)
        incorrect_bias_delta = st.slider("Incorrect bias delta", 0.0, 0.3, 0.05, 0.01,
                                         help="Prefer incorrect_mask if within this delta of the top probability")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found!")
        st.info("Please train the model first using the training script.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model_cached(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing faces to detect masks"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.header("Detection Results")
        
        if uploaded_file is not None:
            # Process image
            with st.spinner("Processing image..."):
                processed_image, predictions_info = detect_masks_in_image(
                    image, model, confidence_threshold,
                    threshold_mask=th_mask,
                    threshold_no_mask=th_no_mask,
                    threshold_incorrect=th_incorrect,
                    return_probs=show_debug,
                    incorrect_bias_delta=incorrect_bias_delta
                )
            
            # Convert BGR to RGB for display
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, caption="Detection Results", use_column_width=True)
            
            # Display predictions
            if predictions_info:
                st.subheader("Detection Details")
                for pred in predictions_info:
                    col_pred, col_conf = st.columns(2)
                    with col_pred:
                        st.write(f"**Face {pred['face_id']}:** {pred['class']}")
                    with col_conf:
                        confidence_color = "green" if pred['confidence'] >= 0.7 else "orange" if pred['confidence'] >= 0.5 else "red"
                        st.markdown(f"<span style='color: {confidence_color}'>Confidence: {pred['confidence']:.3f}</span>", 
                                  unsafe_allow_html=True)
                    if show_debug and 'probs' in pred:
                        st.caption("per-class probabilities")
                        st.json({k: round(v, 3) for k, v in pred['probs'].items()})
            else:
                st.info("No faces detected in the image.")
        else:
            st.info("Please upload an image to see detection results.")
    
    # Footer
    st.markdown("---")
    st.markdown("**FaceMask-Plus** - Advanced Face Mask Detection System")
    st.markdown("Built with TensorFlow, OpenCV, and Streamlit")

if __name__ == "__main__":
    main()

