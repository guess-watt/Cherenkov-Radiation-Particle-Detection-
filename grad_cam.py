import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# ===============================
# SETTINGS

MODEL_PATH = "particle_classifier.h5"
IMG_PATH = "dataset/test/electron/electron_260.png"  # change image if needed
IMG_SIZE = 224
LAST_CONV_LAYER = "block_13_expand"

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)

# ===============================
# LOAD & PREPROCESS IMAGE
# ===============================
img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ===============================
# GRAD-CAM FUNCTION
# ===============================
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(LAST_CONV_LAYER).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ===============================
# OVERLAY HEATMAP
# ===============================
img_original = cv2.imread(IMG_PATH)
img_original = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

heatmap = cv2.resize(heatmap,(IMG_SIZE, IMG_SIZE))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

# ===============================
# DISPLAY RESULTS
# ===============================
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")


plt.show()
