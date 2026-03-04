import os
import sys
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tensorflow import keras
from tensorflow.keras import layers
import uuid
from datetime import datetime
import threading
import time
import base64

# =====================================================
# TERMINAL OUTPUT HELPER
# =====================================================
def log_info(message):
    """Print message to terminal with immediate flush"""
    print(message, flush=True)

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov'}

IMG_SIZE = 224
MARGIN_RATIO = 0.15
NORMAL_THRESHOLD = 0.75
DAMAGED_THRESHOLD = 0.40
BLUR_THRESHOLD = 100

YOLO_MODEL_PATH = r"D:\Nethmyy__Research\Robust-Road-sign-detector-for-PP2\Weights\Detect_Model\RoadSignDetector_v22\weights\best.pt"
CLASS_MODEL_PATH = r"D:\Nethmyy__Research\Robust-Road-sign-detector-for-PP2\Weights\Custom_model2_weights\epoch_026.weights.h5"
CLASS_MAPPING_PATH = r"D:\Nethmyy__Research\Robust-Road-sign-detector-for-PP2\Weights\Custom_model2_weights\class_mapping.json"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# =====================================================
# HELPER: Convert CV2 image to base64 data URI
# =====================================================
def image_to_base64(img):
    """Convert OpenCV image to base64 data URI for HTML display"""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

# =====================================================
# LOAD MODELS
# =====================================================
detector = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO Detection Model Loaded")

with open(CLASS_MAPPING_PATH, "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_indices)

# =====================================================
# BUILD CLASSIFICATION MODEL
# =====================================================
def conv_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def depthwise_block(x, filters, stride=1):
    shortcut = x
    x = layers.DepthwiseConv2D((3,3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (1,1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and shortcut.shape[-1] == filters:
        x = layers.Add()([x, shortcut])

    x = layers.ReLU()(x)
    return x

def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3,3), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128)

    x = depthwise_block(x, 256, stride=2)
    x = depthwise_block(x, 256)

    x = depthwise_block(x, 512, stride=2)
    x = depthwise_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs)

classifier = build_model()
classifier.load_weights(CLASS_MODEL_PATH)
print("✅ Classification Model Loaded")

# =====================================================
# IMAGE QUALITY FUNCTIONS
# =====================================================
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def sharpen_image(image):
    gaussian = cv2.GaussianBlur(image, (9,9), 10)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def classify_crop(crop):
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_norm = crop_rgb.astype("float32") / 255.0
    crop_input = np.expand_dims(crop_norm, axis=0)

    pred = classifier.predict(crop_input, verbose=0)[0]
    class_id = np.argmax(pred)
    class_name = index_to_class[class_id]
    confidence = float(pred[class_id])
    return class_name, confidence

# =====================================================
# PROCESS FRAME
# =====================================================
def process_frame(frame, verbose=True):
    original = frame.copy()
    results = detector(frame, conf=0.25)
    boxes = results[0].boxes

    if len(boxes) == 0:
        if verbose:
            log_info("[INFO] No road sign detected in frame")
        return original, original, None, "No detection", 0, None

    best_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

    w = x2 - x1
    h = y2 - y1
    mx = int(w * MARGIN_RATIO)
    my = int(h * MARGIN_RATIO)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(frame.shape[1], x2 + mx)
    y2 = min(frame.shape[0], y2 + my)

    crop = frame[y1:y2, x1:x2]

    blur_value = blur_score(crop)

    # Original Classification
    class_o, conf_o = classify_crop(crop)

    best_class = class_o
    best_conf = conf_o
    best_crop = crop
    best_method = "Original"

    # Terminal logging for analysis
    if verbose:
        log_info("")
        log_info("=" * 60)
        log_info("ROAD SIGN DETECTION ANALYSIS")
        log_info("=" * 60)
        log_info(f"[INFO] Blur Score: {blur_value:.2f}")
        log_info(f"[Original] {class_o} | {conf_o:.4f}")

    # Enhancements if Blurry
    if blur_value < BLUR_THRESHOLD:
        sharpened = sharpen_image(crop)
        class_s, conf_s = classify_crop(sharpened)

        if verbose:
            log_info(f"[Sharpen]  {class_s} | {conf_s:.4f}")

        if conf_s > best_conf:
            best_conf = conf_s
            best_class = class_s
            best_crop = sharpened
            best_method = "Sharpen"

        clahe_img = apply_clahe(crop)
        class_c, conf_c = classify_crop(clahe_img)

        if verbose:
            log_info(f"[CLAHE]    {class_c} | {conf_c:.4f}")

        if conf_c > best_conf:
            best_conf = conf_c
            best_class = class_c
            best_crop = clahe_img
            best_method = "CLAHE"

    confidence = best_conf
    class_name = best_class
    crop = best_crop

    # Status Decision
    if confidence >= NORMAL_THRESHOLD:
        status = "Normal"
    elif confidence < DAMAGED_THRESHOLD and blur_value < BLUR_THRESHOLD:
        status = "Damaged (Unclear Image)"
    elif confidence < DAMAGED_THRESHOLD and blur_value >= BLUR_THRESHOLD:
        status = "Uncertain Classification"
    else:
        status = "Possibly Unclear"

    # Final result logging
    if verbose:
        log_info(f"[FINAL] {class_name} | {confidence:.4f} | {status}")
        log_info(f"[METHOD] Best result from: {best_method}")
        log_info("=" * 60)

    # Visualization
    if status == "Normal":
        color = (0, 255, 0)
    elif "Damaged" in status:
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)

    detected_img = original.copy()
    cv2.rectangle(detected_img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(detected_img,
                f"{class_name} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2)

    crop_display = crop.copy()
    cv2.putText(crop_display,
                f"{class_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    return original, detected_img, crop_display, class_name, confidence, (x1, y1, x2, y2)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        input_type = request.form.get('input_type')
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + str(uuid.uuid4()) + '_' + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return redirect(url_for('process', filename=filename, input_type=input_type))
    
    return render_template('upload.html')

@app.route('/process/<filename>')
def process(filename):
    input_type = request.args.get('input_type', 'image')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return "File not found", 404
    
    if input_type == 'image':
        img = cv2.imread(filepath)
        if img is None:
            return "Invalid image file", 400
        
        original, detected, crop, class_name, confidence, bbox = process_frame(img)
        
        if crop is None:
            return render_template('no_detection.html')
        
        # Save results
        orig_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_original.jpg")
        detected_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_detected.jpg")
        crop_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_crop.jpg")
        
        cv2.imwrite(orig_path, original)
        cv2.imwrite(detected_path, detected)
        cv2.imwrite(crop_path, crop)
        
        # Convert to relative paths for templates
        orig_rel = orig_path.replace('\\', '/')
        detected_rel = detected_path.replace('\\', '/')
        crop_rel = crop_path.replace('\\', '/')
        
        return render_template('results.html',
                             original_image=orig_rel,
                             detected_image=detected_rel,
                             crop_image=crop_rel,
                             class_name=class_name,
                             confidence=f"{confidence:.2%}",
                             input_type='image')
    
    elif input_type == 'video':
        cap = cv2.VideoCapture(filepath)
        results_list = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0:  # Process every 5th frame
                continue
            
            original, detected, crop, class_name, confidence, bbox = process_frame(frame)
            
            if crop is not None:
                # Save frame results
                orig_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_frame.jpg")
                detected_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_detected.jpg")
                crop_path = os.path.join(RESULTS_FOLDER, f"{uuid.uuid4()}_crop.jpg")
                
                cv2.imwrite(orig_path, original)
                cv2.imwrite(detected_path, detected)
                cv2.imwrite(crop_path, crop)
                
                results_list.append({
                    'frame': frame_count,
                    'original': orig_path.replace('\\', '/'),
                    'detected': detected_path.replace('\\', '/'),
                    'crop': crop_path.replace('\\', '/'),
                    'class_name': class_name,
                    'confidence': f"{confidence:.2%}"
                })
            
            if len(results_list) >= 5:  # Limit to 5 frames for display
                break
        
        cap.release()
        
        if not results_list:
            return render_template('no_detection.html')
        
        return render_template('video_results.html', results=results_list, input_type='video')
    
    return "Invalid input type", 400

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/capture_webcam', methods=['POST'])
def capture_webcam():
    data = request.json
    # This would require base64 encoded image from webcam
    # For now, we'll return a placeholder
    return jsonify({'status': 'success'})

# =====================================================
# VIDEO STREAMING FOR REAL-TIME DETECTION
# =====================================================
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.last_detection = {
            'class_name': None,
            'confidence': 0,
            'bbox': None
        }
        
    def __del__(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()
    
    def get_frame(self, verbose=True):
        success, frame = self.video.read()
        if not success:
            return None, None
        
        # Process frame for detection
        original = frame.copy()
        results = detector(frame, conf=0.25)
        boxes = results[0].boxes
        
        detection_info = {
            'class_name': None,
            'confidence': 0,
            'bbox': None,
            'status': None
        }
        
        if len(boxes) > 0:
            best_box = max(boxes, key=lambda b: float(b.conf[0]))
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
            
            w = x2 - x1
            h = y2 - y1
            mx = int(w * MARGIN_RATIO)
            my = int(h * MARGIN_RATIO)
            
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(frame.shape[1], x2 + mx)
            y2 = min(frame.shape[0], y2 + my)
            
            crop = frame[y1:y2, x1:x2]
            
            # Calculate blur score
            blur_value = blur_score(crop)
            
            # Original classification
            class_o, conf_o = classify_crop(crop)
            
            best_class = class_o
            best_conf = conf_o
            best_method = "Original"
            
            # Terminal logging for webcam
            if verbose:
                log_info("")
                log_info("=" * 60)
                log_info("WEBCAM ROAD SIGN DETECTION")
                log_info("=" * 60)
                log_info(f"[INFO] Blur Score: {blur_value:.2f}")
                log_info(f"[Original] {class_o} | {conf_o:.4f}")
            
            # Apply enhancements if blurry
            if blur_value < BLUR_THRESHOLD:
                sharpened = sharpen_image(crop)
                class_s, conf_s = classify_crop(sharpened)
                
                if verbose:
                    log_info(f"[Sharpen]  {class_s} | {conf_s:.4f}")
                
                if conf_s > best_conf:
                    best_conf = conf_s
                    best_class = class_s
                    best_method = "Sharpen"
                
                clahe_img = apply_clahe(crop)
                class_c, conf_c = classify_crop(clahe_img)
                
                if verbose:
                    log_info(f"[CLAHE]    {class_c} | {conf_c:.4f}")
                
                if conf_c > best_conf:
                    best_conf = conf_c
                    best_class = class_c
                    best_method = "CLAHE"
            
            class_name = best_class
            confidence = best_conf
            
            # Status decision
            if confidence >= NORMAL_THRESHOLD:
                status = "Normal"
            elif confidence < DAMAGED_THRESHOLD and blur_value < BLUR_THRESHOLD:
                status = "Damaged (Unclear Image)"
            elif confidence < DAMAGED_THRESHOLD and blur_value >= BLUR_THRESHOLD:
                status = "Uncertain Classification"
            else:
                status = "Possibly Unclear"
            
            if verbose:
                log_info(f"[FINAL] {class_name} | {confidence:.4f} | {status}")
                log_info(f"[METHOD] Best result from: {best_method}")
                log_info("=" * 60)
            
            # Determine color based on confidence
            if confidence >= NORMAL_THRESHOLD:
                color = (0, 255, 0)  # Green
            elif confidence < DAMAGED_THRESHOLD:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add background for text
            label = f"{class_name} ({confidence:.2%})"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            detection_info = {
                'class_name': class_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'status': status
            }
            self.last_detection = detection_info
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), detection_info
    
    def release(self):
        self.running = False
        if self.video.isOpened():
            self.video.release()

# Global camera instance
camera = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = VideoCamera()
        return camera

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def generate_frames():
    cam = get_camera()
    while True:
        frame, detection_info = cam.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    release_camera()
    return jsonify({'status': 'stopped'})

@app.route('/get_detection_info')
def get_detection_info():
    cam = get_camera()
    return jsonify(cam.last_detection)

@app.route('/process_webcam_frame', methods=['POST'])
def process_webcam_frame():
    """Process a single frame from webcam and return detection result"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        # Process frame
        original, detected, crop, class_name, confidence, bbox = process_frame(frame)
        
        if crop is None:
            return jsonify({
                'detected': False,
                'message': 'No road sign detected'
            })
        
        # Convert to base64 (no disk storage)
        orig_base64 = image_to_base64(original)
        detected_base64 = image_to_base64(detected)
        crop_base64 = image_to_base64(crop)
        
        return jsonify({
            'detected': True,
            'original': orig_base64,
            'detected_image': detected_base64,
            'crop': crop_base64,
            'class_name': class_name,
            'confidence': f"{confidence:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results_from_webcam')
def results_from_webcam():
    """Display results from webcam capture"""
    original = request.args.get('original')
    detected = request.args.get('detected')
    crop = request.args.get('crop')
    class_name = request.args.get('class_name')
    confidence = request.args.get('confidence')
    
    return render_template('results.html',
                         original_image=original,
                         detected_image=detected,
                         crop_image=crop,
                         class_name=class_name,
                         confidence=confidence,
                         input_type='webcam')

if __name__ == '__main__':
    # Ensure unbuffered output for real-time terminal logging
    import os
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    log_info("Starting Road Sign Detection Server...")
    log_info("Server running at: http://localhost:5000")
    log_info("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
