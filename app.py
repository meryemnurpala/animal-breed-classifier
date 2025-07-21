from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import io
import os

app = Flask(__name__)

IMG_SIZE = 224
LIST_FILE = 'data/annotations/list.txt'
MODEL_PATH = 'oxford_pets_model.h5'

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

# Modeli yükle
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = np.array(img_array, dtype=np.float32, copy=True)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    top_breed = class_names[top3_idx[0]]
    top_percentage = int(preds[top3_idx[0]] * 100)
    other_results = []
    for i in top3_idx[1:]:
        other_results.append({
            'breed': class_names[i],
            'percentage': int(preds[i] * 100)
        })
    return jsonify({
        'topBreed': top_breed,
        'topPercentage': top_percentage,
        'otherResults': other_results
    })

if __name__ == '__main__':
    app.run(debug=True) 