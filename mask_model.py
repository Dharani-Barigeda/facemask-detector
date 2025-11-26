import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_mask_detector_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Create a MobileNetV2-based face mask detector model.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (mask, no_mask, incorrect_mask)
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model(model, learning_rate=0.0001):
    """
    Unfreeze the base model for fine-tuning.
    
    Args:
        model: The model to unfreeze
        learning_rate: Learning rate for fine-tuning
    
    Returns:
        Updated model
    """
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

