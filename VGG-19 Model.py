import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Define data directory
data_dir = 'D:/FYP/Train'

# Define image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 16

# Create a VGG-19 base model with pre-trained weights
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a custom top model for binary classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the model
initial_epochs = 20
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=initial_epochs,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Unfreeze the last 4 layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model (necessary after unfreezing layers)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])


# Fine-tune the model
fine_tune_epochs = 10  # Adjust as necessary
total_epochs = initial_epochs + fine_tune_epochs

# Define the validation data directory
validation_data_dir = 'D:/FYP/Validation'

# Data generator for validation set (No augmentation, just rescaling)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=total_epochs,
    initial_epoch=model.history.epoch[-1],  # Start from the last initial epoch
    validation_data=validation_generator,  # Include validation data if available
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Save the fine-tuned model
model.save('cancer_classification_model1.h5')
print("Model fine-tuned and saved successfully")