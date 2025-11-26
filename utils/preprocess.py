import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 512

def preprocess_orbscan_image(file_bytes):
    """Convert uploaded file → numpy array → denoise → resize → normalize."""
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image upload")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    return img
