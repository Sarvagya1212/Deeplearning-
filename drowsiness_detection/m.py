import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2


# --------- Step 1: Synthetic Data Creation (Replace with real data) ---------

IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_SAMPLES = 5000
NUM_CLASSES = 2  # Awake vs Asleep


def generate_synthetic_data(num_samples):
    # Random images (use your real images here)
    images = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)

    # Random one-hot eye state labels
    eye_states = np.zeros((num_samples, NUM_CLASSES))
    random_classes = np.random.choice(NUM_CLASSES, num_samples)
    eye_states[np.arange(num_samples), random_classes] = 1

    # Random ages between 18 and 60
    ages = np.random.uniform(18, 60, (num_samples, 1)).astype(np.float32)

    return images, eye_states, ages


# Create train and val synthetic data
train_images, train_eye_labels, train_ages = generate_synthetic_data(int(NUM_SAMPLES * 0.8))
val_images, val_eye_labels, val_ages = generate_synthetic_data(int(NUM_SAMPLES * 0.2))


# --------- Step 2: Define Multi-output Model ---------

input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

eye_state_output = Dense(NUM_CLASSES, activation='softmax', name='eye_state')(x)
age_output = Dense(1, activation='linear', name='age_pred')(x)

model = Model(inputs=base_model.input, outputs=[eye_state_output, age_output])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        'eye_state': 'categorical_crossentropy',
        'age_pred': 'mean_squared_error'
    },
    metrics={
        'eye_state': ['accuracy'],
        'age_pred': ['mean_absolute_error']
    }
)

print(model.summary())


# --------- Step 3: Train the Model ---------

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_eye_state_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1, monitor='val_eye_state_accuracy'),
    ModelCheckpoint('drowsiness_model.h5', save_best_only=True, monitor='val_eye_state_accuracy', verbose=1)
]

history = model.fit(
    x=train_images,
    y={'eye_state': train_eye_labels, 'age_pred': train_ages},
    validation_data=(val_images, {'eye_state': val_eye_labels, 'age_pred': val_ages}),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)


# --------- Step 4: Simple Inference / Demo on a Random Sample ---------

sample_idx = 0
sample_img = val_images[sample_idx:sample_idx + 1]
pred_eye_state, pred_age = model.predict(sample_img)

predicted_class = np.argmax(pred_eye_state)
eye_state_str = "Awake" if predicted_class == 0 else "Asleep"
predicted_age = float(pred_age[0][0])

print(f"Predicted Eye State: {eye_state_str} (Confidence: {pred_eye_state[0][predicted_class]:.3f})")
print(f"Predicted Age: {predicted_age:.1f} years")
