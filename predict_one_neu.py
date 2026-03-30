import json
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "mobilenetv2_neu_defect_best.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = 224

def load_class_names():
    with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    # {"crazing": 0, "inclusion": 1, ...} -> index 순서대로 정렬
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def main(image_path):
    model = load_model(MODEL_PATH)
    class_names = load_class_names()

    x = preprocess_image(image_path)
    pred = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(pred))
    pred_class = class_names[pred_idx]
    confidence = float(pred[pred_idx])

    print("예측 클래스:", pred_class)
    print("신뢰도:", round(confidence, 4))
    print("전체 확률:", pred)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python predict_one_neu.py test.jpg")
    else:
        main(sys.argv[1])