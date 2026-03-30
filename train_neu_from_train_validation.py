import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# 1. 기본 설정
# =========================
TRAIN_DIR = "train"
VAL_DIR = "validation"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "mobilenetv2_neu_defect_best.keras"
CLASS_INDEX_PATH = "class_indices.json"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 데이터 제너레이터
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",   # 원본이 grayscale이어도 rgb로 읽어서 3채널로 맞춤
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False
)

num_classes = train_generator.num_classes

print("클래스 수:", num_classes)
print("클래스 인덱스:", train_generator.class_indices)

# 클래스 순서 저장
with open(CLASS_INDEX_PATH, "w", encoding="utf-8") as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=2)

# =========================
# 3. MobileNetV2 모델 구성
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# 1차 학습: 백본 고정
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 4. 콜백
# =========================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        mode="max",
        restore_best_weights=True,
        verbose=1
    )
]

# =========================
# 5. 학습
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\n학습 완료")
print("최고 성능 모델 저장:", MODEL_PATH)
print("클래스 인덱스 저장:", CLASS_INDEX_PATH)

# =========================
# 6. 검증 성능 확인
# =========================
val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")