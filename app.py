from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "best.pt"
CLASSES = ['door', 'floor', 'furniture', 'moving_object', 'wall']
OBSTACLE = ['furniture', 'moving_object', 'wall']

# Load model
model = YOLO(MODEL_PATH)

# =====================================================
# CLEAR PATH LOGIC
# =====================================================
def compute_clear_path(mask):
    H, W = mask.shape
    roi = mask[int(0.6 * H):H, :]
    free_cols = [c for c in range(W) if roi[:, c].sum() == 0]

    if not free_cols:
        return None, None, None

    regions, start = [], free_cols[0]

    for i in range(1, len(free_cols)):
        if free_cols[i] != free_cols[i - 1] + 1:
            regions.append((start, free_cols[i - 1]))
            start = free_cols[i]
    regions.append((start, free_cols[-1]))

    best = max(regions, key=lambda x: x[1] - x[0])
    cx = (best[0] + best[1]) // 2

    return cx, best[0], best[1]


def draw_path(frame, cx, left, right):
    H, W, _ = frame.shape
    cv2.rectangle(frame, (left, int(0.6 * H)), (right, H), (0, 255, 0), 2)
    cv2.line(frame, (cx, int(0.6 * H)), (cx, H), (0, 255, 0), 3)
    return frame


# =====================================================
# PROCESS FRAME
# =====================================================
def process_frame(frame):
    H, W, _ = frame.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_name = CLASSES[int(box.cls[0])]
        if cls_name in OBSTACLE:
            mask[y1:y2, x1:x2] = 1

    cx, left, right = compute_clear_path(mask)
    annotated = results.plot()

    frame_center = W // 2

    if cx is None:
        direction = "blocked"
    else:
        annotated = draw_path(annotated, cx, left, right)
        if cx < frame_center - 40:
            direction = "left"
        elif cx > frame_center + 40:
            direction = "right"
        else:
            direction = "straight"

    return annotated, direction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    from flask import request
    
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process frame
        annotated, direction = process_frame(frame)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'direction': direction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)