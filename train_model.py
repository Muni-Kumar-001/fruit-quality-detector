import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Set paths
base_dir = 'C:/Users/DELL/Desktop/project_for_final_year/fruit_quality_detector/dataset'  # Adjust to your dataset path

# Create ImageDataGenerator for training
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # Split data into training and validation
)

# Load training data
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),  # Resize images
    batch_size=32,
    class_mode='binary',  # Binary classification (good/bad)
    subset='training'
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='validation'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Adjust the number of epochs as needed
)

# Save the model
model.save('C:/Users/DELL/Desktop/project_for_final_year/fruit_quality_detector/3rdVersion/models/apple_quality_model.h5')
