import tensorflow as tf
import numpy as np
import argparse
import os

def convert_to_tflite(model_path, output_path=None, quantize=True):
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to the Keras model (.h5 file)
        output_path: Path for the output TFLite model
        quantize: Whether to apply quantization for smaller model size
    
    Returns:
        Path to the converted TFLite model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}.tflite"
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Create a representative dataset for quantization
    def representative_data_gen():
        for _ in range(100):
            # Generate random data in the same format as training data
            data = np.random.random((1, 224, 224, 3)).astype(np.float32)
            yield [data]
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get model size
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {model_size:.2f} MB")
    
    return output_path

def test_tflite_model(tflite_path, test_input_shape=(1, 224, 224, 3)):
    """
    Test the converted TFLite model with random input.
    
    Args:
        tflite_path: Path to the TFLite model
        test_input_shape: Shape of test input
    """
    print("Testing TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Prepare test input
    test_input = np.random.random(test_input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Test output shape: {output.shape}")
    print(f"Test output sample: {output[0][:3]}")  # First 3 values
    
    print("TFLite model test completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite')
    parser.add_argument('--model', type=str, default='best_mask_detector.h5',
                       help='Path to the Keras model file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for TFLite model')
    parser.add_argument('--no_quantize', action='store_true',
                       help='Disable quantization')
    parser.add_argument('--test', action='store_true',
                       help='Test the converted model')
    
    args = parser.parse_args()
    
    try:
        # Convert model
        tflite_path = convert_to_tflite(
            args.model, 
            args.output, 
            not args.no_quantize
        )
        
        # Test model if requested
        if args.test:
            test_tflite_model(tflite_path)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

