from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage import measure
import base64
import os

app = Flask(__name__)
CORS(app)

# Get API key from environment variable
API_KEY = os.environ.get('API_KEY', 'your-default-api-key')

def validate_api_key():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    token = auth_header.split(' ')[1]
    return token == API_KEY

@app.route('/trace', methods=['POST'])
def trace_shapes():
    # Check API key
    if not validate_api_key():
        return jsonify({'error': 'Unauthorized'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    # Read the image in memory
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    original_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the image for both edges and segments
    result = process_image(original_img)

    # Create response including background_base64
    response = {
        'edges': result['edges_base64'],
        'segments': result['segments_base64'],
        'regions': result['regions'],
        'segment_patches': result['segment_patches'],
        'background_base64': result['background_base64'],
        'image_width': original_img.shape[1],
        'image_height': original_img.shape[0]
    }

    return jsonify(response)

def process_image(img):
    """Process image to get edges and segmented regions with memory optimization."""
    # Store original image for final output
    original_img = img.copy()
    
    # Add size check and downscaling for very large images
    max_dimension = 2000  # Maximum dimension we'll process
    scale = 1.0
    if max(img.shape) > max_dimension:
        scale = max_dimension / max(img.shape)
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Multi-scale edge detection approach with memory-efficient parameters
    # 2a. Fine detail edge detection
    bilateral_fine = cv2.bilateralFilter(gray, d=3, sigmaColor=5, sigmaSpace=5)
    edges_fine = cv2.Canny(bilateral_fine, 5, 40)
    bilateral_fine = None  # Free memory
    
    # 2b. Normal edge detection
    bilateral_normal = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    edges_normal = cv2.Canny(bilateral_normal, 15, 90)
    bilateral_normal = None  # Free memory
    
    # 2c. Combine edge detection results
    edges_combined = cv2.bitwise_or(edges_fine, edges_normal)
    edges_fine = None  # Free memory
    edges_normal = None  # Free memory
    
    # 3. Edge enhancement with smaller kernel
    cross_kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    edges_connected = cv2.dilate(edges_combined, cross_kernel, iterations=1)
    edges_combined = None  # Free memory
    
    # 4. Create binary mask
    binary = edges_connected > 0
    edges_connected = None  # Free memory
    
    # 5. Connected components with size filtering
    num_labels, labels = cv2.connectedComponents(
        (1 - binary).astype(np.uint8),
        connectivity=4  # Use 4-connectivity instead of 8 to reduce complexity
    )
    
    # Filter out very small components to reduce processing
    min_component_size = 100
    for label in range(1, num_labels):
        component_size = np.sum(labels == label)
        if component_size < min_component_size:
            labels[labels == label] = 0
    
    # Reassign labels to be consecutive
    unique_labels = np.unique(labels)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    new_labels = np.zeros_like(labels)
    for old_label, new_label in label_map.items():
        new_labels[labels == old_label] = new_label
    labels = new_labels
    num_labels = len(unique_labels)
    
    # Generate colors
    colors = []
    for i in range(num_labels):
        hue = int((i / num_labels) * 180)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(rgb_color)
    
    colors = np.array(colors)
    colors[0] = [0, 0, 0]  # Background color
    
    # Create visualization efficiently
    segments_with_edges = np.zeros_like(img)
    for label in range(num_labels):
        mask = (labels == label)
        segments_with_edges[mask] = colors[label]
    
    segments_with_edges[binary] = [0, 0, 0]
    
    edge_output = img.copy()
    edge_output[binary] = [255, 0, 255]
    
    # Process segments more efficiently
    segment_patches = []
    for label in range(1, num_labels):
        try:
            mask = (labels == label)
            if not np.any(mask):
                continue
            
            # Use numpy operations instead of where()
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            top, bottom = np.where(rows)[0][[0, -1]]
            left, right = np.where(cols)[0][[0, -1]]
            
            # Skip if patch is too large
            if (bottom - top + 1) * (right - left + 1) > 1000000:  # Skip patches larger than 1M pixels
                continue
                
            patch = original_img[top:bottom+1, left:right+1].copy()
            patch_mask = mask[top:bottom+1, left:right+1]
            
            patch_with_alpha = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
            edge_mask = binary[top:bottom+1, left:right+1]
            expanded_mask = patch_mask | edge_mask
            
            patch_with_alpha[~expanded_mask] = [0, 0, 0, 0]
            
            _, buffer = cv2.imencode('.png', patch_with_alpha)
            patch_base64 = base64.b64encode(buffer).decode('utf-8')
            
            segment_patches.append({
                'label': int(label),
                'base64': patch_base64,
                'position': {
                    'top': int(top),
                    'left': int(left),
                    'width': int(right - left + 1),
                    'height': int(bottom - top + 1)
                }
            })
        except Exception as e:
            continue  # Skip problematic segments
    
    # Create background
    background = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
    _, bg_buffer = cv2.imencode('.png', background)
    background_base64 = base64.b64encode(bg_buffer).decode('utf-8')
    
    # Calculate regions more efficiently
    props = measure.regionprops(labels)
    regions = []
    min_area = 100
    
    for prop in props:
        if prop.label > 0 and prop.area >= min_area:
            regions.append({
                'label': int(prop.label),
                'area': int(prop.area),
                'centroid': [int(prop.centroid[1]), int(prop.centroid[0])],
                'bbox': [int(x) for x in prop.bbox],
                'equivalent_diameter': float(prop.equivalent_diameter)
            })
    
    _, edges_buffer = cv2.imencode('.png', edge_output)
    _, segments_buffer = cv2.imencode('.png', segments_with_edges)
    
    return {
        'edges_base64': base64.b64encode(edges_buffer).decode('utf-8'),
        'segments_base64': base64.b64encode(segments_buffer).decode('utf-8'),
        'segment_patches': segment_patches,
        'regions': regions,
        'background_base64': background_base64
    }
    
@app.route('/')
def home():
    return "Flask server is running!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
