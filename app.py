import os
import time
import base64
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'models/yolov8_export.onnx'

# Load model once at startup if available
try:
    model = YOLO(MODEL_PATH)
    print(f'Loaded YOLO model from {MODEL_PATH}')
except Exception as e:
    model = None
    print(f'Could not load model from {MODEL_PATH}: {e}')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(file_path)

    if model is None:
        return jsonify({'error': 'Model not loaded. Please export YOLO ONNX model first.'}), 500

    results = model(file_path, conf=0.25, iou=0.45)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{image_file.filename}')
    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)

    return jsonify({
        'original': url_for('uploaded_file', filename=image_file.filename),
        'annotated': url_for('uploaded_file', filename=f'result_{image_file.filename}')
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/live', methods=['GET'])
def live():
    return render_template('live.html')


@app.route('/live-frame', methods=['POST'])
def live_frame():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please export YOLO ONNX model first.'}), 500

    data = request.get_json()
    if data is None or 'image' not in data:
        return jsonify({'error': 'No image data received.'}), 400

    image_data = data['image'].split(',')[-1]
    image_bytes = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_data), np.uint8),
        cv2.IMREAD_COLOR
    )
    if image_bytes is None:
        return jsonify({'error': 'Could not decode image data.'}), 400

    results = model(image_bytes, conf=0.25, iou=0.45)
    annotated = results[0].plot()
    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'annotated': f'data:image/jpeg;base64,{annotated_b64}'})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'model_loaded': model is not None, 'model_path': MODEL_PATH})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
