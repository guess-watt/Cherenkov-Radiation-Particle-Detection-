import os
import cv2
import numpy as np
import random

IMG_SIZE = 224
IMAGES_PER_CLASS = 300
BASE_DIR = "dataset"

classes = {
    "electron": {"blur": 11, "noise": 40, "thickness": 6, "brightness": 1.3},
    "muon": {"blur": 3, "noise": 10, "thickness": 8, "brightness": 1.0},
    "pion": {"blur": 7, "noise": 25, "thickness": 7, "brightness": 1.1},
    "proton": {"blur": 5, "noise": 20, "thickness": 4, "brightness": 0.9}
}

splits = ["train", "validation", "test"]

def make_dirs():
    for split in splits:
        for cls in classes:
            os.makedirs(f"{BASE_DIR}/{split}/{cls}", exist_ok=True)

def add_motion_blur(img):
    if random.random() > 0.5:
        kernel_size = random.choice([3,5,7])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        img = cv2.filter2D(img, -1, kernel)
    return img

def generate_ring(params):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Random center shift
    center_x = IMG_SIZE//2 + random.randint(-20, 20)
    center_y = IMG_SIZE//2 + random.randint(-20, 20)
    center = (center_x, center_y)

    # Random radius
    radius = random.randint(50, 90)

    thickness = params["thickness"] + random.randint(-2, 2)

    # Random partial ring
    if random.random() > 0.7:
        start_angle = random.randint(0, 180)
        end_angle = start_angle + random.randint(180, 320)
        cv2.ellipse(img, center, (radius, radius), 0,
                    start_angle, end_angle,
                    (255,255,255), thickness)
    else:
        cv2.circle(img, center, radius, (255,255,255), thickness)

    # Gaussian blur
    img = cv2.GaussianBlur(img, (params["blur"], params["blur"]), 0)

    # Gaussian noise
    noise = np.random.normal(0, params["noise"], img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Brightness scaling
    img = cv2.convertScaleAbs(img, alpha=params["brightness"] + random.uniform(-0.2, 0.2))

    # Motion blur
    img = add_motion_blur(img)

    # Random rotation
    angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))

    return img

# Create folders
make_dirs()

# Generate dataset
for cls, params in classes.items():
    for i in range(IMAGES_PER_CLASS):
        img = generate_ring(params)

        if i < 200:
            split = "train"
        elif i < 250:
            split = "validation"
        else:
            split = "test"

        cv2.imwrite(f"{BASE_DIR}/{split}/{cls}/{cls}_{i}.png", img)

print("Realistic dataset generation completed.")