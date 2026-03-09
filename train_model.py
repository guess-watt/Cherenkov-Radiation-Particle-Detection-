import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# ===============================
# SETTINGS

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
DATASET_PATH = "dataset"

# ===============================
# DATA GENERATORS (STRONG AUGMENTATION)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.3],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===============================
# LOAD PRETRAINED MODEL
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ===============================
# ADD CUSTOM HEAD
# ===============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)   # Increased dropout
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ===============================
# COMPILE MODEL (LOW LR IMPORTANT)
# ===============================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# EARLY STOPPING
# ===============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================
# TRAIN MODEL
# ===============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("model", exist_ok=True)
model.save("model/model.h5")
model.save("particle_classifier.h5")

print("Model training completed and saved.")

# ===============================
# PLOT RESULTS
# ===============================
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")

plt.savefig("training_plot.png")
