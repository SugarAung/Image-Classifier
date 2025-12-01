import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# DATA PATHS
train_dir = 'C:/Project/SSET/datasets/train'
valid_dir = 'C:/Project/SSET/datasets/valid'
test_dir = 'C:/Project/SSET/datasets/test'

# AUGMENTED TRAIN GENERATOR (for generalization)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# VALIDATION & TEST GENERATORS (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# LOAD DATA
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(75, 75),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(75, 75),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(75, 75),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# CALCULATE CLASS WEIGHTS
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))

# MODEL DEFINITION
model = Sequential([
    Input(shape=(75, 75, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes
])

# COMPILE MODEL
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAINING
history = model.fit(
    train_generator,
    epochs=13,
    validation_data=valid_generator,
    class_weight=class_weight_dict
)

# SAVE MODEL (new format recommended)
model.save('sset_model.keras')
print(" Model saved as sset_model.keras")

# PLOTS - ACCURACY
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# PLOTS - LOSS
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
