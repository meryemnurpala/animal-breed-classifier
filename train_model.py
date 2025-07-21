import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'data/images'
ANNOTATION_FILE = 'data/annotations/trainval.txt'
LIST_FILE = 'data/annotations/list.txt'

# Sınıf isimlerini list.txt'den çıkar
class_id_to_name = {}
with open(LIST_FILE, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name, class_id = parts[0], int(parts[1])
        if class_id not in class_id_to_name:
            breed = '_'.join(img_name.split('_')[:-1])
            class_id_to_name[class_id] = breed
        if len(class_id_to_name) == 37:
            break
class_names = [class_id_to_name[i+1] for i in range(37)]

# Görüntü ve etiketleri oku
image_paths = []
labels = []
with open(ANNOTATION_FILE, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        img_name, class_id = parts[0], int(parts[1])
        img_path = os.path.join(DATA_DIR, img_name + '.jpg')
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(class_id - 1)  # 0-indexed

# Eğitim/Doğrulama ayrımı
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Modeli oluştur
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(37, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('oxford_pets_model.h5', save_best_only=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
) 