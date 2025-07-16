
# 🧠 NeuroScan AI

**NeuroScan AI** is a web-based application for automated brain tumor detection using MRI scans. Built on the YOLOv8 object detection model, it provides real-time predictions, confidence scores, and visual heatmaps to support medical diagnosis.

## 🔍 Project Overview

This project combines deep learning and web development to enable fast, accessible tumor detection with a focus on:
- Glioma
- Meningioma
- Pituitary tumors

It features:
- Real-time MRI image analysis
- Visual annotations and Grad-CAM heatmaps
- Interactive history tracking
- PDF/JSON export of results

## 🛠️ Technologies Used

**Backend:**
- Python
- Flask
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV

**Frontend:**
- HTML/CSS
- Bootstrap 5
- Chart.js

**AI Tools:**
- YOLOv8 custom-trained model
- Grad-CAM for explainable AI

## 🧪 Dataset

Dataset sourced from [Kaggle – Brain Tumor MRI Detection](https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-detection/data)  
Includes labeled MRI scans across four categories: glioma, meningioma, pituitary, and no tumor. Images are annotated with bounding boxes and formatted for YOLOv8 training.

## 🧠 Model Training

- Custom YOLOv8-nano model fine-tuned on the dataset
- Image size: 640×640
- Three output classes with bounding boxes and confidence levels
- Training performed with PyTorch on CPU (for demo purposes)

## 📂 Project Structure

```
neuroscan-ai/
├── static/
│   ├── uploads/
│   ├── results/
│   ├── heatmaps/
├── templates/
│   └── index.html
├── yolov8/
│   └── best.pt
├── upload_history.json
└── app.py
```

## 🚀 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/YassirB808/NeuroScan-AI
   cd neuroscan-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
