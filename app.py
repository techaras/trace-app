from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage import measure
import base64
import os
from concurrent.futures import ThreadPoolExecutor
import gc

app = Flask(__name__)
CORS(app)

# Constants for image processing
MAX_IMAGE_DIMENSION = 2000
CHUNK_SIZE = 512  # Size for tiled processing
MAX_WORKERS = 2  # Match CPU count

def validate_api_key():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    token = auth_header.split(' ')[1]
    return token == API_KEY

def resize_if_needed(img):
    """Resize image if it exceeds maximum dimensions while maintaining aspect ratio"""
    height, width = img.shape[:2]
    max_dim = max(height, width)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def process_tile(args):
    """Process a single tile of the image"""
    tile, params = args
    if tile.size == 0:
        return None
        
    # Local bilateral filtering and edge detection
    bilateral = cv2.bilateralFilter(tile, d=3, sigmaColor=10, sigmaSpace=10)
    edges = cv2.Canny(bilateral, 10, 60)
    return edges

def process_image(img):
    """Optimized image processing with tiled processing and memory management"""
    try:
        # Initial resize if needed
        img = resize_if_needed(img)
        original_img = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        del img  # Free memory
        gc.collect()
        
        # Prepare tiles for processing
        height, width = gray.shape
        tiles = []
        params = []
        
        for y in range(0, height, CHUNK_SIZE):
            for x in range(0, width, CHUNK_SIZE):
                tile = gray[y:min(y + CHUNK_SIZE, height), 
                          x:min(x + CHUNK_SIZE, width)]
                tiles.append(tile)
                params.append({})  # Parameters for processing if needed
        
        # Process tiles in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            edge_tiles = list(executor.map(process_tile, zip(tiles, params)))
        
        # Reconstruct edge image
        edges = np.zeros_like(gray)
        tile_idx = 0
        for y in range(0, height, CHUNK_SIZE):
            for x in range(0, width, CHUNK_SIZE):
                if edge_tiles[tile_idx] is not None:
                    h, w = edge_tiles[tile_idx].shape
                    edges[y:y+h, x:x+w] = edge_tiles[tile_idx]
                tile_idx += 1
        
        # Clear tiles from memory
        del tiles, edge_tiles
        gc.collect()
        
        # Create binary mask
        binary = edges > 0
        
        # Connected components with memory-efficient processing
        labels = np.zeros_like(gray, dtype=np.int32)
        num_labels, labels = cv2.connectedComponents(
            (1 - binary).astype(np.uint8),
            connectivity=8
        )
        
        # Generate colors efficiently
        colors = np.zeros((num_labels, 3), dtype=np.uint8)
        for i in range(num_labels):
            hue = int((i / num_labels) * 180)
            hsv_color = np.uint8([[[hue, 255, 255]]])
            colors[i] = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        colors[0] = [0, 0, 0]  # Background color
        
        # Create segments efficiently
        segments_with_edges = np.zeros_like(original_img)
        for label in range(num_labels):
            mask = labels == label
            segments_with_edges[mask] = colors[label]
        segments_with_edges[binary] = [0, 0, 0]
        
        # Process segments with optimized memory usage
        segment_patches = []
        for label in range(1, num_labels):
            mask = labels == label
            if not np.any(mask):
                continue
            
            # Get bounds
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            
            # Extract patch with minimal memory usage
            top, bottom = np.min(y_indices), np.max(y_indices)
            left, right = np.min(x_indices), np.max(x_indices)
            
            patch = original_img[top:bottom+1, left:right+1].copy()
            patch_mask = mask[top:bottom+1, left:right+1]
            
            # Create patch with alpha channel
            patch_with_alpha = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
            patch_with_alpha[~patch_mask] = [0, 0, 0, 0]
            
            # Encode patch
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
            
            # Clear patch memory
            del patch, patch_with_alpha, buffer
            gc.collect()
        
        # Process background
        background = cv2.cvtColor(original_img, cv2.COLOR_BGR2BGRA)
        _, bg_buffer = cv2.imencode('.png', background)
        background_base64 = base64.b64encode(bg_buffer).decode('utf-8')
        
        # Generate region properties
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
        
        # Create final output
        edge_output = original_img.copy()
        edge_output[binary] = [255, 0, 255]
        
        _, edges_buffer = cv2.imencode('.png', edge_output)
        _, segments_buffer = cv2.imencode('.png', segments_with_edges)
        
        return {
            'edges_base64': base64.b64encode(edges_buffer).decode('utf-8'),
            'segments_base64': base64.b64encode(segments_buffer).decode('utf-8'),
            'segment_patches': segment_patches,
            'regions': regions,
            'background_base64': background_base64
        }
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise

@app.route('/trace', methods=['POST'])
def trace_shapes():
    try:
        if not validate_api_key():
            return jsonify({'error': 'Unauthorized'}), 401

        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 400

        # Read and process image
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        original_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if original_img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        result = process_image(original_img)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
    