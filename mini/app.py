# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.activations import swish
from efficientnet.tfkeras import EfficientNetB0  
from collections import defaultdict
from sklearn.cluster import KMeans
from PIL import Image
import base64
import io

def build_model_architecture() -> tf.keras.Model:
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(64, 64, 3))
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(64, 64, 3)),
        base_model,
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

def load_eurosat_model(model_path: str) -> tf.keras.Model:
    # Try loading full SavedModel with custom objects first
    try:
        try:
            # Prefer the efficientnet package's swish/FixedDropout if available
            try:
                from efficientnet import model as effnet_model
                custom_objects = {
                    'swish': getattr(effnet_model, 'swish', tf.nn.swish),
                    'FixedDropout': getattr(effnet_model, 'FixedDropout', tf.keras.layers.Dropout),
                    'EfficientNetB0': EfficientNetB0,
                }
            except Exception:
                custom_objects = {
                    'swish': tf.nn.swish,
                    'EfficientNetB0': EfficientNetB0,
                }
            return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception:
            # Fallback: rebuild architecture and load weights
            model = build_model_architecture()
            try:
                model.load_weights(model_path)
                return model
            except Exception:
                # Last resort: try loading without any custom objects
                return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

model = load_eurosat_model('model.keras')


CLASS_NAMES = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# =====================================================
# CHANGE DETECTION FUNCTIONS
# =====================================================

def preprocess_image_for_change_detection(image_file):
    """Read and preprocess image from file upload for change detection"""
    img_bytes = image_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_kmeans_clustering(image, n_clusters=5):
    """Apply K-means clustering to segment the image"""
    # Reshape image to 2D array of pixels
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image shape
    clustered = labels.reshape(h, w)
    return clustered, kmeans.cluster_centers_

def detect_changes_kmeans(image1, image2, n_clusters=5, threshold=0.3):
    """
    Detect changes between two images using K-means clustering
    
    Args:
        image1: First image (before)
        image2: Second image (after)
        n_clusters: Number of clusters for K-means
        threshold: Threshold for change detection (0-1)
    
    Returns:
        change_map: Binary map showing changed regions
        change_percentage: Percentage of changed pixels
        visualization: Colored visualization of changes
    """
    # Resize images to same size if needed
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    if (h1, w1) != (h2, w2):
        # Resize to smaller dimensions
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        image1 = cv2.resize(image1, (target_w, target_h))
        image2 = cv2.resize(image2, (target_w, target_h))
    
    print(f"Change detection: Processing images of size {image1.shape}")
    
    # Apply K-means clustering to both images
    print("Applying K-means clustering to image 1...")
    clusters1, centers1 = apply_kmeans_clustering(image1, n_clusters)
    print("Applying K-means clustering to image 2...")
    clusters2, centers2 = apply_kmeans_clustering(image2, n_clusters)
    
    # Calculate difference between cluster assignments
    change_map = (clusters1 != clusters2).astype(np.uint8)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    change_map = cv2.morphologyEx(change_map, cv2.MORPH_OPEN, kernel)
    change_map = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    
    # Calculate change percentage
    total_pixels = change_map.size
    changed_pixels = np.sum(change_map)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    print(f"Change percentage: {change_percentage:.2f}%")
    
    # Create visualization
    # Convert original image2 to grayscale for base
    gray_base = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    gray_base = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2RGB)
    
    # Create colored overlay for changes
    visualization = gray_base.copy()
    
    # Highlight changes in red
    visualization[change_map == 1] = [255, 0, 0]
    
    # Blend with original image
    alpha = 0.6
    visualization = cv2.addWeighted(image2, alpha, visualization, 1-alpha, 0)
    
    # Add contours around changed regions
    contours, _ = cv2.findContours(change_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
    
    print(f"Found {len(contours)} change regions")
    
    return change_map, change_percentage, visualization

def image_to_base64(image_array):
    """Convert numpy array to base64 string"""
    # Convert RGB to BGR for OpenCV encoding
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        image_bgr = (image_array * 255).astype(np.uint8)
    
    # Encode to PNG
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64
    img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return img_str

# =====================================================
# END CHANGE DETECTION FUNCTIONS
# =====================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for API access

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api')
def api_info():
    return jsonify({
        "message": "EuroSAT Land Use Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/predict",
            "annotate": "/api/annotate",
            "change-detection": "/api/change-detection",
            "health": "/api/health"
        },
        "usage": "Send POST requests with image files to /api/predict, /api/annotate, or /api/change-detection"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict land use class for a single image patch.
    
    Request:
        - POST with 'file' field containing image
        
    Response:
        - JSON with predicted class and confidence
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please include an image file in the request'
        }), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Please upload an image file (PNG, JPG, JPEG, TIFF, BMP)'
        }), 400
    
    try:
        img = preprocess_image(file.read())
        preds = model.predict(img, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return jsonify({
            'success': True,
            'data': {
            'class': CLASS_NAMES[class_idx],
                'class_id': int(class_idx),
                'confidence': round(confidence * 100, 2),
                'confidence_raw': float(confidence),
                'all_predictions': {
                    CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 2) 
                    for i in range(len(CLASS_NAMES))
                }
            },
            'timestamp': str(np.datetime64('now')),
            'model_info': {
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'classes': CLASS_NAMES
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process image'
        }), 500

CLASS_COLORS = {
    'AnnualCrop': (255, 0, 0),      # Red
    'Forest': (0, 255, 0),          # Green
    'HerbaceousVegetation': (173, 216, 230),  # Light Blue
    'Highway': (255, 255, 0),       # Yellow
    'Industrial': (255, 0, 255),    # Magenta
    'Pasture': (0, 255, 255),       # Cyan
    'PermanentCrop': (221, 160, 221), # Light Purple
    'Residential': (128, 128, 0),   # Olive
    'River': (0, 128, 128),         # Teal
    'SeaLake': (255, 182, 193),     # Light Pink
}

def draw_label_with_bg_and_arrow(image, text, position, bg_color, dot_center, font_scale=0.9, thickness=3):
    """Draws a label with a colored background, black text, and an arrow pointing to the label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = position
    bg_x1, bg_y1 = x - 5, y - text_h - 5
    bg_x2, bg_y2 = x + text_w + 5, y + 5

    # Draw background rectangle
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)

    # Draw black text
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    # Draw an arrow from the dot center to the label
    cv2.arrowedLine(image, dot_center, (x + text_w // 2, bg_y1), bg_color, 2)

@app.route('/api/annotate', methods=['POST'])
def annotate():
    """
    Perform sliding window annotation on a large image.
    
    Request:
        - POST with 'file' field containing image
        
    Response:
        - JSON with annotated image, scores, and metadata
    """
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'message': 'Please include an image file in the request'
        }), 400
    
    file = request.files['file']
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return jsonify({
            'error': 'Invalid file type',
            'message': 'Please upload an image file (PNG, JPG, JPEG, TIFF, BMP)'
        }), 400
    
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        input_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if input_image_bgr is None:
            return jsonify({
                'error': 'Unable to decode image',
                'message': 'The uploaded file could not be processed as an image'
            }), 400
        
        input_image = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = input_image.shape

        patch_size = 64
        stride = 32

        overlay = input_image.copy()
        class_detections = defaultdict(list)

        # Slide over image
        for y in range(0, max(height - patch_size + 1, 1), stride):
            for x in range(0, max(width - patch_size + 1, 1), stride):
                patch = input_image[y:y + patch_size, x:x + patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    # Skip incomplete edge patches
                    continue
                patch_float = patch.astype(np.float32) / 255.0
                patch_input = np.expand_dims(patch_float, axis=0)
                prediction = model.predict(patch_input, verbose=0)
                predicted_label = int(np.argmax(prediction, axis=1)[0])
                confidence = float(np.max(prediction)) * 100.0
                class_detections[predicted_label].append((x, y, confidence))

        output_order = ['Highway', 'AnnualCrop', 'Industrial', 'River', 'Residential']
        scores = {}
        for class_name in output_order:
            if class_name in CLASS_NAMES:
                class_label = CLASS_NAMES.index(class_name)
                if class_label in class_detections and class_detections[class_label]:
                    max_conf = max(d[2] for d in class_detections[class_label])
                    scores[class_name] = round(max_conf, 1)
                else:
                    scores[class_name] = 0.0
            else:
                scores[class_name] = 0.0

        # Visualization: draw best point per detected class
        print(f"Debug: Found {len(class_detections)} classes with detections")
        for class_label, detections in class_detections.items():
            if not detections:
                continue
            best_detection = max(detections, key=lambda d: d[2])
            x, y, confidence = best_detection
            class_name = CLASS_NAMES[class_label]
            color = CLASS_COLORS[class_name]
            print(f"Debug: Drawing {class_name} at ({x}, {y}) with confidence {confidence:.1f}%")

            # Draw a larger dot at the center of the patch
            cx = x + patch_size // 2
            cy = y + patch_size // 2
            cv2.circle(overlay, (cx, cy), 15, color, -1)  # Even larger dot size for visibility
            # Add a black border to make dots more visible
            cv2.circle(overlay, (cx, cy), 15, (0, 0, 0), 2)

            # Draw label with background and arrow
            label_text = f"{class_name}: {confidence:.1f}%"
            
            # Smart label positioning to keep it within image bounds
            label_x = cx + 20  # Default offset
            label_y = cy - 20  # Default offset
            
            # Adjust if label would go outside image boundaries
            if label_x + 150 > width:  # Approximate text width
                label_x = cx - 170  # Move label to the left
            if label_y < 30:  # Too close to top
                label_y = cy + 20  # Move label below the dot
            elif label_y > height - 30:  # Too close to bottom
                label_y = cy - 20  # Keep label above the dot
                
            draw_label_with_bg_and_arrow(overlay, label_text, (label_x, label_y), color, (cx, cy))

        # Encode annotated image to base64 PNG
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.png', overlay_bgr)
        if not success:
            print("Failed to encode PNG image")
            return jsonify({
                'error': 'Failed to encode annotated image',
                'message': 'Could not process the annotated image'
            }), 500
        
        b64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        print(f"Successfully encoded image. Base64 length: {len(b64_image)}")
        print(f"Image dimensions: {overlay.shape}")

        # Overall top-1 prediction on center crop (quick preview)
        center_y = max((height - patch_size) // 2, 0)
        center_x = max((width - patch_size) // 2, 0)
        center_patch = input_image[center_y:center_y+patch_size, center_x:center_x+patch_size]
        center_patch = center_patch.astype(np.float32) / 255.0
        center_pred = model.predict(np.expand_dims(center_patch, 0), verbose=0)
        top_idx = int(np.argmax(center_pred, axis=1)[0])
        top_conf = float(np.max(center_pred))

        response_data = {
            'success': True,
            'data': {
                'top_prediction': {
                    'class': CLASS_NAMES[top_idx],
                    'class_id': int(top_idx),
                    'confidence': round(top_conf * 100, 2),
                    'confidence_raw': float(top_conf)
                },
                'class_scores': scores,
                'annotated_image': b64_image,
                'image_metadata': {
                    'width': width,
                    'height': height,
                    'patches_analyzed': len([(y, x) for y in range(0, max(height - patch_size + 1, 1), stride) 
                                           for x in range(0, max(width - patch_size + 1, 1), stride)])
                }
            },
            'timestamp': str(np.datetime64('now')),
            'model_info': {
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'classes': CLASS_NAMES
            }
        }
        
        print(f"Response data keys: {list(response_data['data'].keys())}")
        print(f"Annotated image present: {'annotated_image' in response_data['data']}")
        print(f"Annotated image length: {len(b64_image) if b64_image else 0}")
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to process image annotation'
        }), 500

# =====================================================
# CHANGE DETECTION ENDPOINT
# =====================================================

@app.route('/api/change-detection', methods=['POST'])
def change_detection():
    """
    API endpoint for change detection between two satellite images.
    
    Request:
        - POST with 'image1' (before) and 'image2' (after) fields containing images
        
    Response:
        - JSON with change detection visualization and statistics
    """
    try:
        # Check if both images are provided
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'error': 'Both image1 and image2 are required',
                'message': 'Please upload both before and after images'
            }), 400
        
        image1_file = request.files['image1']
        image2_file = request.files['image2']
        
        # Validate file types
        for img_file, name in [(image1_file, 'image1'), (image2_file, 'image2')]:
            if not img_file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                return jsonify({
                    'error': f'Invalid file type for {name}',
                    'message': 'Please upload image files (PNG, JPG, JPEG, TIFF, BMP)'
                }), 400
        
        print("Starting change detection...")
        
        # Preprocess images
        print("Loading image 1...")
        image1 = preprocess_image_for_change_detection(image1_file)
        print(f"Image 1 loaded: {image1.shape}")
        
        print("Loading image 2...")
        image2 = preprocess_image_for_change_detection(image2_file)
        print(f"Image 2 loaded: {image2.shape}")
        
        # Perform change detection using K-means clustering
        print("Performing change detection with K-means clustering...")
        change_map, change_percentage, visualization = detect_changes_kmeans(
            image1, image2, n_clusters=5, threshold=0.3
        )
        
        # Convert visualization to base64
        print("Encoding result image...")
        change_image_base64 = image_to_base64(visualization)
        print(f"Result image encoded. Base64 length: {len(change_image_base64)}")
        
        response_data = {
            'success': True,
            'change_image': change_image_base64,
            'change_percentage': round(change_percentage, 2),
            'data': {
                'change_image': change_image_base64,
                'change_percentage': round(change_percentage, 2),
                'total_pixels': int(change_map.size),
                'changed_pixels': int(np.sum(change_map)),
                'image_dimensions': {
                    'width': int(visualization.shape[1]),
                    'height': int(visualization.shape[0])
                }
            },
            'message': f'Change detection completed. {round(change_percentage, 2)}% of the area has changed.',
            'timestamp': str(np.datetime64('now')),
            'algorithm': {
                'method': 'K-means Clustering',
                'n_clusters': 5,
                'description': 'Detects changes by comparing K-means cluster assignments between images'
            }
        }
        
        print("Change detection completed successfully!")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in change detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to perform change detection'
        }), 500

# =====================================================
# END CHANGE DETECTION ENDPOINT
# =====================================================

@app.route('/api/test-image', methods=['GET'])
def test_image():
    """
    Test endpoint to verify image processing and base64 encoding.
    """
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:, :, 0] = 255  # Red channel
        
        # Add some text
        cv2.putText(test_img, 'TEST', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode to base64
        success, buffer = cv2.imencode('.png', test_img)
        if not success:
            return jsonify({'error': 'Failed to encode test image'}), 500
        
        b64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'test_image': b64_image,
            'message': 'Test image generated successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    
    Response:
        - JSON with service status and model info
    """
    try:
        # Test model with a dummy input
        dummy_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
        return jsonify({
            'status': 'healthy',
            'service': 'EuroSAT Land Use Classification API',
            'timestamp': str(np.datetime64('now')),
            'model': {
                'status': 'loaded',
                'name': 'EfficientNetB0',
                'input_shape': [64, 64, 3],
                'num_classes': len(CLASS_NAMES),
                'classes': CLASS_NAMES
            },
            'features': {
                'classification': True,
                'annotation': True,
                'change_detection': True
            },
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': str(np.datetime64('now'))
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("EuroSAT Land Use Classification API")
    print("=" * 60)
    print("Available endpoints:")
    print("  - GET  /              : Serve index.html")
    print("  - GET  /api           : API information")
    print("  - POST /api/predict   : Single image classification")
    print("  - POST /api/annotate  : Sliding window annotation")
    print("  - POST /api/change-detection : Change detection between two images")
    print("  - GET  /api/health    : Health check")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)