NOTE: this repo does not contain the used datasets, becuase combined size of datasets is >5GB and it will take more than a day on out internet. Same with the fused dataset

# Intelligent Threat Detection System

An advanced machine learning-based threat detection system using YOLOv8 object detection to identify and classify military and airborne objects in images and live video streams.

## Project Overview

This project implements a real-time threat detection system capable of recognizing 7 distinct object classes:
- **Airplane** - Fixed-wing aircraft
- **Helicopter** - Rotary-wing aircraft  
- **Drone** - Unmanned aerial vehicles
- **Bird** - Natural flying objects
- **Tank** - Ground-based military vehicles
- **Person** - Individual human subjects
- **Soldier** - Armed personnel

The system combines deep learning inference with a Flask web application for accessible threat monitoring and analysis.

## Features

- **Real-time Detection**: Live video stream threat detection with YOLO inference
- **Image Analysis**: Upload and analyze static images for threats
- **Multiple Model Support**: YOLOv8 and YOLOv2 pre-trained weights included
- **ONNX Export**: Optimized model export for cross-platform deployment
- **Web Interface**: User-friendly Flask-based web application
- **Rest API**: JSON-based API endpoints for detection and system status

## Requirements

- Python 3.8+
- Flask
- OpenCV (cv2)
- NumPy
- PyTorch
- TorchVision
- Ultralytics YOLO
- ONNX & ONNX Runtime
- scikit-learn
- Matplotlib

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PremadeSCo/ML-FINAL_PROJECT.git
cd ML-FINAL_PROJECT
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
ML-FINAL_PROJECT/
├── app.py                      # Flask web application
├── threat_detection.ipynb      # Main Jupyter notebook with model training & analysis
├── military_objects.yaml       # YOLO dataset configuration
├── requirements.txt            # Python dependencies
├── yolov8s.pt                 # YOLOv8 Small pre-trained weights (22MB)
├── yolo26n.pt                 # YOLOv2 6-class pre-trained weights (5.5MB)
└── uploads/                    # Directory for uploaded images & results
```

## Usage

### Running the Web Application

```bash
python app.py
```

The Flask application will start on `http://localhost:5000`

**Available Routes:**
- `/` - Home page with image upload interface
- `/upload` - POST endpoint for image threat detection
- `/live` - Live video stream threat detection
- `/live-frame` - POST endpoint for real-time frame processing
- `/uploads/<filename>` - Retrieve processed images
- `/status` - GET endpoint returning model status

### Using the Jupyter Notebook

Open `threat_detection.ipynb` to:
- Train custom models on your dataset
- Evaluate model performance
- Visualize detection results
- Export models to ONNX format

## 🔧 Configuration

### Model Path
Edit `app.py` to specify the YOLO model:
```python
MODEL_PATH = 'models/yolov8_export.onnx'  # Or use .pt weights directly
```

### Detection Parameters
Adjust confidence and IoU thresholds in `app.py`:
```python
results = model(file_path, conf=0.25, iou=0.45)
```

### Dataset Configuration
Edit `military_objects.yaml` to customize:
- Dataset paths
- Number of classes
- Class names
- Train/validation splits

## Dataset Classes

The model is trained to detect 7 distinct threat categories:

| Class | Description |
|-------|-------------|
| Airplane | Fixed-wing military or commercial aircraft |
| Helicopter | Rotary-wing aircraft and variants |
| Drone | Unmanned aerial vehicles (UAV) |
| Bird | Flying birds (differentiation from threats) |
| Tank | Ground-based armored vehicles |
| Person | Individual human subjects |
| Soldier | Armed or uniformed personnel |

## Model Information

- **Architecture**: YOLOv8 (Small variant)
- **Framework**: PyTorch + Ultralytics
- **Input**: RGB images (variable size)
- **Output**: Bounding boxes with class labels and confidence scores
- **Export Formats**: PyTorch (.pt), ONNX (.onnx)

## API Endpoints

### Upload Image for Analysis
```http
POST /upload
Content-Type: multipart/form-data

image: <file>
```

**Response:**
```json
{
  "original": "/uploads/image.jpg",
  "annotated": "/uploads/result_image.jpg"
}
```

### Check Model Status
```http
GET /status
```

**Response:**
```json
{
  "model_loaded": true,
  "model_path": "models/yolov8_export.onnx"
}
```

### Live Frame Detection
```http
POST /live-frame
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,..."
}
```

## Training Your Own Model

1. Prepare your dataset following YOLO format
2. Update `military_objects.yaml` with your dataset paths
3. Open `threat_detection.ipynb` and run training cells
4. Export trained model to ONNX format
5. Update `MODEL_PATH` in `app.py`

## Performance Notes

- **Inference Speed**: ~50-100ms per frame (GPU dependent)
- **Supported Resolutions**: 416x416 to 1280x1280
- **Confidence Threshold**: 0.25 (adjustable)
- **IoU Threshold**: 0.45 (adjustable)

## License

This project is provided as-is for educational and research purposes.

## Author

**PremadeSCo**
**Bilal**
**Abduu**

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: June 2026  
**Status**: Active Development
