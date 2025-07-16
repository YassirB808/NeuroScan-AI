from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
import time
from datetime import datetime
from ultralytics import YOLO
import json
import pdfkit
from fpdf import FPDF
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
HEATMAP_FOLDER = 'static/heatmaps'
HISTORY_FILE = 'upload_history.json'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8/best.pt')

# Tumor information database
TUMOR_INFO = {
    'glioma': {
        'description': 'Gliomas are tumors that develop from glial cells in the brain and spinal cord. They can be benign or malignant.',
        'symptoms': 'Headaches, nausea, vomiting, seizures, memory loss, personality changes',
        'treatment': 'Surgery, radiation therapy, chemotherapy'
    },
    'meningioma': {
        'description': 'Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord. Most are benign.',
        'symptoms': 'Headaches, vision problems, hearing loss, memory difficulties',
        'treatment': 'Observation, surgery, radiation therapy'
    },
    'pituitary': {
        'description': 'Pituitary tumors are abnormal growths in the pituitary gland. Most are benign adenomas.',
        'symptoms': 'Headaches, vision loss, hormonal imbalances',
        'treatment': 'Medication, surgery, radiation therapy'
    }
}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activation = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]
    
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activation).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

def preprocess_image(file_path):
    image = cv2.imread(file_path)
    resized = cv2.resize(image, (640, 640))
    return resized

def generate_gradcam(image_path, model, target_layer):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    
    grad_cam = GradCAM(model=model, target_layer=target_layer)
    heatmap = grad_cam(img_tensor)
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def generate_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="NeuroScan AI - Brain Tumor Detection Report", ln=1, align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Scan Information", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(200, 10, txt=f"Original Filename: {data['original_filename']}", ln=1)
    pdf.ln(5)
    
    # Results Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Detection Results", ln=1)
    pdf.set_font("Arial", size=12)
    
    if data['detections']:
        for i, detection in enumerate(data['detections'], 1):
            pdf.cell(200, 10, txt=f"Tumor {i}: {detection['class']} ({detection['confidence']}%)", ln=1)
            pdf.cell(200, 10, txt=f"Location: {detection['box']}", ln=1)
            pdf.ln(3)
    else:
        pdf.cell(200, 10, txt="No tumors detected", ln=1)
    
    pdf.ln(10)
    
    # Add the result image (centered and smaller)
    result_image_path = os.path.join(RESULT_FOLDER, data['result_img'])
    if os.path.exists(result_image_path):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Analysis Results", ln=1, align='C')
        pdf.ln(8)
        
        # Calculate centered position and reduced size
        img_width = 100  # Reduced from 190 (about 25% smaller)
        x_position = (210 - img_width) / 2  # 210mm is A4 width
        
        # Add image with centered alignment and reduced size
        pdf.image(result_image_path, x=x_position, w=img_width)
        
        pdf.ln(10)
    
    # Model Information
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Model Information", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Processing Time: {data['prediction_time']} ms", ln=1, align='C')
    pdf.cell(200, 10, txt="Model: YOLOv8 Brain Tumor Detection", ln=1, align='C')
    
    # Create and return BytesIO buffer
    buffer = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer

@app.route('/', methods=['GET', 'POST'])
def index():
    history = load_history()
    conf_threshold = float(request.args.get('conf', 0.5))
    
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='No file selected.', 
                                current_year=datetime.now().year,
                                history=history,
                                conf_threshold=conf_threshold)

        filename = secure_filename(file.filename)
        original_filename = filename.rsplit('.', 1)[0]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_name = f"{original_filename}_{timestamp}.jpg"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(file_path)

        image = preprocess_image(file_path)
        start_time = time.time()
        results = model.predict(source=image, imgsz=640, conf=conf_threshold)
        end_time = time.time()
        prediction_time = round((end_time - start_time) * 1000, 2)

        annotated = results[0].plot()
        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated)

        heatmap_filename = f"heatmap_{timestamp}.jpg"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        demo_heatmap = cv2.applyColorMap(cv2.resize(image, (640, 640)), cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, demo_heatmap)

        detections = []
        if results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = round(float(box.conf[0]) * 100, 2)
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': [round(x, 2) for x in box.xyxy[0].tolist()]
                })

        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original_filename': filename,
            'uploaded_img': unique_name,
            'result_img': result_filename,
            'heatmap_img': heatmap_filename,
            'prediction_time': prediction_time,
            'detections': detections
        }
        history.insert(0, history_entry)
        save_history(history[:10])

        return render_template('index.html',
                           result_img=result_filename,
                           heatmap_img=heatmap_filename,
                           uploaded_img=unique_name,
                           prediction_time=prediction_time,
                           detections=detections,
                           current_year=datetime.now().year,
                           history=history,
                           conf_threshold=conf_threshold,
                           tumor_info=TUMOR_INFO)

    return render_template('index.html', 
                         current_year=datetime.now().year,
                         history=history,
                         conf_threshold=conf_threshold)

@app.route('/export/pdf/<filename>')
def export_pdf(filename):
    history = load_history()
    entry = next((item for item in history if item['result_img'] == filename), None)
    
    if not entry:
        return "Report not found", 404
    
    pdf_buffer = generate_pdf_report(entry)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"NeuroScan_Report_{filename.split('.')[0]}.pdf"
    )

@app.route('/export/json/<filename>')
def export_json(filename):
    history = load_history()
    entry = next((item for item in history if item['result_img'] == filename), None)
    
    if not entry:
        return "Report not found", 404
    
    return jsonify(entry)

if __name__ == '__main__':
    app.run(debug=True)