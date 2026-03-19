from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from gradcam_utils import make_gradcam

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("model/model.h5")

# MobileNetV2 last conv layer
LAST_CONV_LAYER = "Conv_1"

# Class labels (must match training order)
CLASS_NAMES = ["electron", "muon","pion", "proton"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(img_rgb / 255.0, axis=0)

        # Prediction
        preds = model.predict(img_array, verbose=0)
        class_index = np.argmax(preds[0])
        confidence = float(np.max(preds[0])) * 100
        predicted_class = CLASS_NAMES[class_index]

        

        # Grad-CAM
        heatmap = make_gradcam(model, img_array, LAST_CONV_LAYER)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save outputs
        heatmap_path = os.path.join(RESULT_FOLDER, "heatmap.jpg")
        overlay_path = os.path.join(RESULT_FOLDER, "overlay.jpg")

        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(overlay_path, overlay)

        return render_template(
            "index.html",
            original=img_path,
            heatmap=heatmap_path,
            overlay=overlay_path,
            predicted_class=predicted_class,
            confidence=round(confidence, 2)
        )

    return render_template("index.html")

if __name__ == "__main__":

    app.run(debug=True)
