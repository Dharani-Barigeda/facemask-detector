import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from mask_model import create_mask_detector_model, unfreeze_model
import json

def prepare_data_generators(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Prepare data generators for training and validation.
    
    Args:
        data_dir: Path to the dataset directory
        img_size: Target image size
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        Tuple of (train_generator, validation_generator, class_indices)
    """
    # If dataset is already split into train/val subdirectories, use them directly
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        validation_generator = validation_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, validation_generator, train_generator.class_indices

    # Fallback: single directory with validation_split
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator, train_generator.class_indices

def train_model(data_dir, epochs=50, batch_size=32, img_size=(224, 224)):
    """
    Train the face mask detector model.
    
    Args:
        data_dir: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Target image size
    
    Returns:
        Trained model and training history
    """
    print("Preparing data generators...")
    train_gen, val_gen, class_indices = prepare_data_generators(
        data_dir, img_size, batch_size
    )
    
    print(f"Class indices: {class_indices}")
    
    # Save class indices for later use
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    print("Creating model...")
    model = create_mask_detector_model(img_size + (3,), len(class_indices))
    
    # Print model summary
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'best_mask_detector.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Compute class weights to handle class imbalance
    print("Computing class weights from training data...")
    class_indices_inv = {v: k for k, v in class_indices.items()}
    y_classes = train_gen.classes
    unique_classes = np.unique(y_classes)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_classes
    )
    class_weight = {int(cls): float(w) for cls, w in zip(unique_classes, class_weights_array)}
    print(f"Class weights: { {class_indices_inv[i]: round(w, 3) for i, w in class_weight.items()} }")

    print("Starting training...")
    # Phase 1: Train only the head
    history1 = model.fit(
        train_gen,
        epochs=epochs//2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
    )
    
    print("Fine-tuning with unfrozen base model...")
    # Phase 2: Fine-tune with unfrozen base
    model = unfreeze_model(model, learning_rate=0.0001)
    
    history2 = model.fit(
        train_gen,
        epochs=epochs//2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight
    )
    
    # Combine histories
    history = {
        'loss': history1.history['loss'] + history2.history['loss'],
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
    }
    
    return model, history

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot and save training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, data_dir, img_size=(224, 224), batch_size=32):
    """
    Evaluate the model and generate detailed metrics.
    
    Args:
        model: Trained model
        data_dir: Path to the dataset directory
        img_size: Target image size
        batch_size: Batch size for evaluation
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Load class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Classification report
    class_names = list(class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Face Mask Detector')
    parser.add_argument('--data_dir', type=str, default='dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224, 
                       help='Image size for training')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found!")
        print("Please create the dataset directory with the following structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── mask/")
        print("│   ├── no_mask/")
        print("│   └── incorrect_mask/")
        exit(1)
    
    # Train the model
    model, history = train_model(
        args.data_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, args.data_dir, (args.img_size, args.img_size))
    
    print("Training completed! Model saved as 'best_mask_detector.h5'")

