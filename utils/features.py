import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from skimage.feature import graycomatrix, graycoprops

# Load CNN once (global)
EFF_MODEL = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(512, 512, 3),
    pooling="avg"
)


def extract_cnn_features(img):
    x = np.expand_dims(img, axis=0)
    feats = EFF_MODEL.predict(x, verbose=0)[0]
    return feats   # (1280,)


def extract_classical_features(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
    feats = [
        graycoprops(gcm, 'contrast')[0,0],
        graycoprops(gcm, 'correlation')[0,0],
        graycoprops(gcm, 'dissimilarity')[0,0],
        graycoprops(gcm, 'homogeneity')[0,0],
        graycoprops(gcm, 'ASM')[0,0],
        graycoprops(gcm, 'energy')[0,0],
        np.mean(gray), np.std(gray), np.min(gray), np.max(gray)
    ]

    return np.array(feats, dtype=np.float32)
