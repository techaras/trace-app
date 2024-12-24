from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage import measure
import base64

app = Flask(__name__)
CORS(app)

@app.route('/trace', methods=['POST'])
def trace_shapes():
    if 'image' not in request.files:
        return 'No image part in the request', 400

    file = request.files['image']
    if file.filename == '':
        return 'No image selected for uploading', 400

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
    """Process image to get edges and segmented regions with improved fine edge detection."""
    # Store original image for final output
    original_img = img.copy()
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Multi-scale edge detection approach
    edges_fine = None
    edges_normal = None
    
    # 2a. Fine detail edge detection
    # Even lighter bilateral filter for finer details
    bilateral_fine = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=10)  # Further reduced parameters
    edges_fine = cv2.Canny(bilateral_fine, 10, 60)  # Even lower thresholds for finest edges
    
    # 2b. Normal edge detection (kept for stability)
    bilateral_normal = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)  # Slightly reduced
    edges_normal = cv2.Canny(bilateral_normal, 20, 110)  # Lower thresholds while maintaining stability
    
    # 2c. Combine edge detection results
    edges_combined = cv2.bitwise_or(edges_fine, edges_normal)
    
    # 3. Careful edge enhancement
    # Use a modified cross-shaped kernel for better detail preservation
    cross_kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=np.uint8)
    
    edges_connected = cv2.dilate(edges_combined, cross_kernel, iterations=1)
    
    # 4. Create binary mask
    binary = edges_connected > 0
    
    # 5. Connected components labeling with 8-connectivity
    num_labels, labels = cv2.connectedComponents(
        (1 - binary).astype(np.uint8),
        connectivity=8
    )
    
    # Create visualization
    segments = np.zeros_like(img)
    
    # Generate colors with HSV for better distinction
    colors = []
    for i in range(num_labels):
        hue = int((i / num_labels) * 180)
        hsv_color = np.uint8([[[hue, 255, 255]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(rgb_color)
    
    colors = np.array(colors)
    colors[0] = [0, 0, 0]  # Background color
    
    segments_with_edges = np.zeros_like(img)
    for label in range(num_labels):
        mask = labels == label
        segments[mask] = colors[label]
        segments_with_edges[mask] = colors[label]
    
    segments_with_edges[binary] = [0, 0, 0]
    
    edge_output = img.copy()
    edge_output[binary] = [255, 0, 255]  # Magenta edges
    
    # Create segment patches
    segment_patches = []
    for label in range(1, num_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
        
        top, bottom = np.min(y_indices), np.max(y_indices)
        left, right = np.min(x_indices), np.max(x_indices)
        
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
    
    background = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
    _, bg_buffer = cv2.imencode('.png', background)
    background_base64 = base64.b64encode(bg_buffer).decode('utf-8')
    
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

