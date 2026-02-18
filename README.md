# Oral Health Detection using YOLOv11

A machine learning project that uses YOLOv11 for automated oral health detection and analysis through object detection.

## 📋 Overview

This project leverages YOLOv11, the latest version of the YOLO (You Only Look Once) object detection model, to identify and analyze various oral health conditions from dental images. The system can detect and classify different dental anomalies and health indicators automatically.

## ✨ Features

- **Real-time Detection**: Fast and accurate oral health anomaly detection using YOLOv11
- **Object Detection**: Identifies multiple dental health indicators in a single image
- **Easy Integration**: Simple API for inference and predictions
- **High Accuracy**: Leverages state-of-the-art YOLOv11 architecture

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/amal5434/Oral-health-Detection-using-yolov11.git
cd Oral-health-Detection-using-yolov11
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model (if not included):
```bash
python -c "from ultralytics import YOLO; YOLO('yolov11n.pt')"
```

## 📖 Usage

### Basic Inference

```python
from ultralytics import YOLO

# Load the model
model = YOLO('path/to/model.pt')

# Run inference on an image
results = model.predict(source='path/to/image.jpg', conf=0.5)

# Display results
for result in results:
    result.show()
```

### Running on Images or Videos

```bash
python detect.py --source path/to/image.jpg --model path/to/model.pt
```

## 📁 Project Structure

```
Oral-health-Detection-using-yolov11/
├── README.md
├── requirements.txt
├── detect.py
├── models/
│   └── (trained models)
├── data/
│   ├── images/
│   └── labels/
└── results/
```

## 📊 Model Information

- **Model**: YOLOv11
- **Task**: Object Detection
- **Framework**: Ultralytics

## 🚀 Performance

| Metric | Value |
|--------|-------|
| Model Size | - |
| Inference Speed | - |
| mAP Score | - |

*Update with your actual metrics*

## 📝 Requirements

See `requirements.txt` for detailed dependencies. Key packages include:
- ultralytics
- opencv-python
- torch
- torchvision

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Last Updated**: February 18, 2026