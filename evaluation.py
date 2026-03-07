import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("particle_classifier.h5")
# (Use this because I can see this file in your folder)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Test data generator (NO augmentation here)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "dataset/test",      # your test folder
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Number of test images:", test_generator.samples)
print("Class labels:", test_generator.class_indices)

# Evaluate
loss, accuracy = model.evaluate(test_generator)
print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))
print("Test Loss:", loss)

# Predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(
    test_generator.classes,
    y_pred,
    target_names=test_generator.class_indices.keys()
))