import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Define dataset path
file_path = "dataset_blood_group"

# Get class names
name_class = os.listdir(file_path)

# Function to get image paths
def get_image_paths(file_path):
    image_paths = []
    for class_dir in name_class:
        class_path = os.path.join(file_path, class_dir)
        if os.path.isdir(class_path):
            image_files = glob.glob(os.path.join(class_path, "*.BMP"))  # Adjust for your image format
            image_paths.extend(image_files)
    return image_paths

image_paths = get_image_paths(file_path)

# Create a DataFrame for images and labels
labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
data = pd.DataFrame({'imagepath': image_paths, 'Label': labels})

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Split data
train, test = train_test_split(data, test_size=0.25, random_state=42)

# Image Data Generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train, x_col='imagepath', y_col='Label',
    target_size=(256, 256), class_mode='categorical', batch_size=32, shuffle=True
)
class_indices = train_gen.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)
valid_gen = test_datagen.flow_from_dataframe(
    dataframe=test, x_col='imagepath', y_col='Label',
    target_size=(256, 256), class_mode='categorical', batch_size=32, shuffle=False
)

# Load Pretrained Model
base_model = ResNet50(input_shape=(256, 256, 3), include_top=False, weights='imagenet', pooling='avg')
base_model.trainable = False  # Freeze model layers

# Add Custom Layers
x = Dense(128, activation='relu')(base_model.output)
x = Dense(64, activation='relu')(x)
outputs = Dense(len(name_class), activation='softmax')(x)  # Adjust for classes

model = Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# class_indices = train_gen.class_indices
# with open("class_indices.json", "w") as f:
#     json.dump(class_indices, f)

# Train Model
#history = model.fit(train_gen, validation_data=valid_gen, epochs=5)
history = model.fit(train_gen, validation_data=valid_gen, epochs=5)

# Save Model
model.save("blood_group_model.h5")
#print("Generator Class Indices:", train_gen.class_indices)
