from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import traceback

app = Flask(__name__)

# Enable CORS for all routes and handle preflight OPTIONS requests
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Define fixed sizes for furniture classes (width, height in meters)
furniture_sizes = {
    'bed': {'large': (1.5, 1.2), 'medium': (1.2, 1.0), 'small': (1.0, 0.8)},
    'chair': {'large': (0.5, 0.5), 'medium': (0.4, 0.4), 'small': (0.3, 0.3)},
    'cupboard': {'large': (1.2, 0.5), 'medium': (1.0, 0.4), 'small': (0.8, 0.3)},
    'drawers': {'large': (1.0, 0.4), 'medium': (0.8, 0.3), 'small': (0.6, 0.3)},
    'fireplace': {'medium': (1.2, 0.4)},
    'ottoman': {'medium': (0.5, 0.5)},
    'table': {'small': (0.6, 0.4)},
    'sofa': {'large': (2.0, 0.8), 'medium': (1.6, 0.7), 'small': (1.2, 0.6)},
    'windows': {'medium': (1.0, 0.1)},  # Adjusted size
    'door': {'medium': (0.8, 0.1)},     # Adjusted size
    # Add more classes as needed
}

# List of classes to ignore (decorative or structural elements)
ignored_classes = [
    'floor', 'walls', 'wallHanging', 'rug', 'curtains',
    'lamp', 'light', 'plant', 'mirror'  # Classes to ignore
]

# Mapping model class names to our classes (if necessary)
class_name_mapping = {
    'sofa': 'sofa',
    'couch': 'sofa',
    'chair': 'chair',
    'bed': 'bed',
    'table': 'table',
    'diningtable': 'table',
    'tvmonitor': 'tv',
    'door': 'door',
    'window': 'windows',
    'fireplace': 'fireplace',
    # Add more mappings based on your model's classes
}

# Initialize the YOLO model (make sure 'model_5.pt' is accessible)
model = YOLO('ai_models/model_5.pt')  # Replace with your trained model or 'yolov5s.pt'

@app.route('/ai/process_image', methods=['POST', 'OPTIONS'])
def process_image():
    if request.method == 'OPTIONS':
        # Handle preflight CORS request
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    try:
        # Check if 'image' is in request.files
        if 'image' not in request.files:
            return jsonify({'error': 'No image file in request'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image into cv2
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Unable to read image file'}), 400

        # Resize the image to 640x640 pixels if it's not already
        image = cv2.resize(image, (640, 640))

        # Perform object detection
        results = model(image)

        # Extract detection results
        detected_objects = []
        for result in results:
            boxes = result.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy()  # x_min, y_min, x_max, y_max

            for cls_id, conf, bbox in zip(class_ids, confidences, bboxes):
                class_name = model.names[cls_id]
                class_name = class_name_mapping.get(class_name, class_name)
                if class_name in ignored_classes:
                    continue
                detected_objects.append({
                    'class': class_name,
                    'bbox': bbox,
                    'confidence': conf
                })

        # If no objects detected, return error
        if not detected_objects:
            return jsonify({'error': 'No relevant objects detected.'}), 400

        # Identify the largest object among furniture as the reference object
        furniture_objects = [obj for obj in detected_objects if obj['class'] not in ['door', 'windows'] + ignored_classes]
        if not furniture_objects:
            return jsonify({'error': 'No furniture objects detected.'}), 400

        largest_object = max(
            furniture_objects,
            key=lambda obj: (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
        )

        # Assign fixed size to the largest object
        reference_class = largest_object['class']
        reference_bbox = largest_object['bbox']
        reference_area = (reference_bbox[2] - reference_bbox[0]) * (reference_bbox[3] - reference_bbox[1])

        # Assume the largest size category for the reference object
        reference_size_category = 'large'
        reference_size = furniture_sizes.get(reference_class, {'medium': (1.0, 1.0)}).get(reference_size_category, (1.0, 1.0))

        # Calculate scaling factor (meters per pixel)
        reference_pixel_width = reference_bbox[2] - reference_bbox[0]
        reference_real_width = reference_size[0]  # Width in meters
        scaling_factor = reference_real_width / reference_pixel_width  # meters per pixel

        # Map object positions and sizes to the layout
        layout_objects = []
        for obj in detected_objects:
            obj_class = obj['class']
            bbox = obj['bbox']
            pixel_width = bbox[2] - bbox[0]
            pixel_height = bbox[3] - bbox[1]
            pixel_area = pixel_width * pixel_height

            # For doors and windows, assign fixed size category
            if obj_class in ['door', 'windows']:
                size_category = 'medium'
            else:
                # Calculate the area ratio compared to the reference object
                area_ratio = pixel_area / reference_area

                # Assign size category based on area ratio
                if area_ratio > 0.7:
                    size_category = 'large'
                elif area_ratio > 0.4:
                    size_category = 'medium'
                else:
                    size_category = 'small'

            # Get real-world size (width, height in meters)
            real_size = furniture_sizes.get(obj_class, {'medium': (1.0, 1.0)}).get(size_category, (1.0, 1.0))

            # Calculate the center position in pixels
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Convert center position to meters using the scaling factor
            real_x = center_x * scaling_factor
            real_y = center_y * scaling_factor

            layout_objects.append({
                'class': obj_class,
                'size': real_size,
                'position': (real_x, real_y),
                'bbox': bbox,
                'size_category': size_category,
                'original_center': (center_x, center_y)  # Store original center position
            })

        # Function to estimate room size based on object positions and sizes
        def estimate_room_size(layout_objects, margin=0.5):
            # Exclude ignored classes from room size estimation
            relevant_objects = [obj for obj in layout_objects if obj['class'] not in ignored_classes]
            # Find the minimum and maximum extents of the objects
            min_x = min(obj['position'][0] - obj['size'][0] / 2 for obj in relevant_objects)
            max_x = max(obj['position'][0] + obj['size'][0] / 2 for obj in relevant_objects)
            min_y = min(obj['position'][1] - obj['size'][1] / 2 for obj in relevant_objects)
            max_y = max(obj['position'][1] + obj['size'][1] / 2 for obj in relevant_objects)

            # Calculate room dimensions with margin
            room_width = max_x - min_x + margin
            room_height = max_y - min_y + margin

            return room_width, room_height, min_x - margin / 2, min_y - margin / 2

        # Estimate room size
        room_width, room_height, offset_x, offset_y = estimate_room_size(layout_objects, margin=0.5)

        # Ensure minimum room size (e.g., 3m x 3m)
        room_width = max(3.0, room_width)
        room_height = max(3.0, room_height)

        room_size_meters = (room_width, room_height)
        canvas_size_pixels = (600, int(600 * (room_height / room_width)))  # Adjust canvas height based on aspect ratio

        # Define grid parameters
        cell_size = 0.5  # Grid cell size in meters for finer placement
        grid_cols = int(np.ceil(room_width / cell_size))
        grid_rows = int(np.ceil(room_height / cell_size))

        # Initialize grid with None
        grid = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]

        # Create a blank image for the layout
        layout_image = np.ones((canvas_size_pixels[1], canvas_size_pixels[0], 3), dtype=np.uint8) * 255  # White background

        # Scaling factors for the layout canvas
        layout_scaling_factor_x = canvas_size_pixels[0] / room_width
        layout_scaling_factor_y = canvas_size_pixels[1] / room_height

        # Draw walls (edges of the room)
        cv2.rectangle(layout_image, (0, 0), (canvas_size_pixels[0]-1, canvas_size_pixels[1]-1), (0, 0, 0), 2)  # Black walls

        # Function to map real-world coordinates to grid indices
        def map_to_grid(x, y):
            grid_x = int(x // cell_size)
            grid_y = int(y // cell_size)
            return grid_x, grid_y

        # Function to check if the required grid cells are free
        def is_grid_free(grid, start_x, start_y, cells_needed_x, cells_needed_y):
            for i in range(start_y, start_y + cells_needed_y):
                for j in range(start_x, start_x + cells_needed_x):
                    if i >= grid_rows or j >= grid_cols:
                        return False
                    if grid[i][j] is not None:
                        return False
            return True

        # Function to mark grid cells as occupied
        def mark_grid(grid, obj, start_x, start_y, cells_needed_x, cells_needed_y):
            for i in range(start_y, start_y + cells_needed_y):
                for j in range(start_x, start_x + cells_needed_x):
                    grid[i][j] = obj

        # Collect windows and doors for special handling
        windows = [obj for obj in layout_objects if obj['class'] == 'windows']
        doors = [obj for obj in layout_objects if obj['class'] == 'door']
        furniture = [obj for obj in layout_objects if obj['class'] not in ['windows', 'door'] + ignored_classes]

        # Function to place windows based on their image position
        def place_windows(windows, grid, room_width, room_height, cell_size):
            image_width = 640  # Your image width
            for window in windows:
                # Get the original center x-coordinate in pixels
                center_x = window['original_center'][0]
                # Determine if the window is closer to the left or right side of the image
                if center_x < image_width * 0.33:
                    closest_wall = 'left'
                elif center_x > image_width * 0.66:
                    closest_wall = 'right'
                else:
                    closest_wall = 'left'  # Default to left if in the middle third

                # Swap dimensions if on left or right wall
                real_width, real_height = window['size']
                if closest_wall in ['left', 'right']:
                    real_width, real_height = real_height, real_width  # Swap dimensions

                # Adjust position to be on the wall line
                if closest_wall == 'left':
                    adjusted_x = 0  # On the left wall line
                    adjusted_y = window['position'][1] - offset_y  # Keep original y
                elif closest_wall == 'right':
                    adjusted_x = room_width  # On the right wall line
                    adjusted_y = window['position'][1] - offset_y  # Keep original y
                else:
                    adjusted_x = 0
                    adjusted_y = room_height / 2  # Center vertically

                # Update window's adjusted position and size
                window['adjusted_position'] = (adjusted_x, adjusted_y)
                window['size'] = (real_width, real_height)

                # Map to grid
                grid_x, grid_y = map_to_grid(adjusted_x, adjusted_y)
                cells_needed_x = max(1, int(np.ceil(real_width / cell_size)))
                cells_needed_y = max(1, int(np.ceil(real_height / cell_size)))

                if is_grid_free(grid, grid_x, grid_y, cells_needed_x, cells_needed_y):
                    mark_grid(grid, window, grid_x, grid_y, cells_needed_x, cells_needed_y)
                else:
                    print(f"Warning: Could not place window '{window['class']}' due to overlap.")

        # Function to place doors based on their positions
        def place_doors(doors, grid, room_width, room_height, cell_size):
            for door in doors:
                obj_class = door['class']
                real_width, real_height = door['size']
                real_x, real_y = door['position']

                # Determine the closest wall
                distances_to_walls = {
                    'left': real_x - offset_x,
                    'right': room_width - (real_x - offset_x),
                    'top': real_y - offset_y,
                    'bottom': room_height - (real_y - offset_y)
                }
                closest_wall = min(distances_to_walls, key=distances_to_walls.get)

                # Adjust position to be on the wall line
                if closest_wall == 'left':
                    adjusted_x = 0
                    adjusted_y = real_y - offset_y
                elif closest_wall == 'right':
                    adjusted_x = room_width
                    adjusted_y = real_y - offset_y
                elif closest_wall == 'top':
                    adjusted_x = real_x - offset_x
                    adjusted_y = 0
                elif closest_wall == 'bottom':
                    adjusted_x = real_x - offset_x
                    adjusted_y = room_height

                # Update door's adjusted position and size
                door['adjusted_position'] = (adjusted_x, adjusted_y)
                door['size'] = (real_width, real_height)

                # Map to grid
                grid_x, grid_y = map_to_grid(adjusted_x, adjusted_y)
                cells_needed_x = max(1, int(np.ceil(real_width / cell_size)))
                cells_needed_y = max(1, int(np.ceil(real_height / cell_size)))

                if is_grid_free(grid, grid_x, grid_y, cells_needed_x, cells_needed_y):
                    mark_grid(grid, door, grid_x, grid_y, cells_needed_x, cells_needed_y)
                else:
                    print(f"Warning: Could not place door '{door['class']}' due to overlap.")

        # Function to place furniture based on their positions
        def place_furniture(furniture, grid, room_width, room_height, cell_size):
            for obj in furniture:
                obj_class = obj['class']
                real_width, real_height = obj['size']
                real_x, real_y = obj['position']

                # Adjust positions based on room offset
                adjusted_x = real_x - offset_x
                adjusted_y = real_y - offset_y

                # Map to grid
                grid_x, grid_y = map_to_grid(adjusted_x, adjusted_y)
                cells_needed_x = max(1, int(np.ceil(real_width / cell_size)))
                cells_needed_y = max(1, int(np.ceil(real_height / cell_size)))

                # Check if object is near the center to adjust placement
                center_threshold_x = room_width * 0.3 <= adjusted_x <= room_width * 0.7
                center_threshold_y = room_height * 0.3 <= adjusted_y <= room_height * 0.7

                if center_threshold_x and center_threshold_y:
                    # Place near the top wall by adjusting the y-coordinate
                    adjusted_y = real_height / 2  # Slightly away from the wall

                    # Re-map to grid after adjustment
                    grid_x, grid_y = map_to_grid(adjusted_x, adjusted_y)

                # Check if placement is within grid bounds
                if grid_x + cells_needed_x > grid_cols or grid_y + cells_needed_y > grid_rows:
                    print(f"Warning: {obj_class} at position ({adjusted_x}, {adjusted_y}) exceeds room boundaries.")
                    continue

                if is_grid_free(grid, grid_x, grid_y, cells_needed_x, cells_needed_y):
                    mark_grid(grid, obj, grid_x, grid_y, cells_needed_x, cells_needed_y)
                    obj['adjusted_position'] = (adjusted_x, adjusted_y)
                else:
                    # Attempt to find nearby free cells while maintaining relative positions
                    placed = False
                    search_radius = 1
                    while not placed and search_radius < max(grid_rows, grid_cols):
                        for dy in range(-search_radius, search_radius + 1):
                            for dx in range(-search_radius, search_radius + 1):
                                new_grid_x = grid_x + dx
                                new_grid_y = grid_y + dy

                                if (0 <= new_grid_x < grid_cols - cells_needed_x + 1 and
                                    0 <= new_grid_y < grid_rows - cells_needed_y + 1):
                                    if is_grid_free(grid, new_grid_x, new_grid_y, cells_needed_x, cells_needed_y):
                                        mark_grid(grid, obj, new_grid_x, new_grid_y, cells_needed_x, cells_needed_y)
                                        # Update adjusted position based on new grid placement
                                        new_real_x = (new_grid_x + cells_needed_x / 2) * cell_size
                                        new_real_y = (new_grid_y + cells_needed_y / 2) * cell_size
                                        obj['adjusted_position'] = (new_real_x, new_real_y)
                                        placed = True
                                        break
                        if placed:
                            break
                        search_radius += 1
                    if not placed:
                        print(f"Warning: Could not place {obj_class} at ({adjusted_x}, {adjusted_y}) due to overlap.")

        # Place windows, doors, and furniture
        place_windows(windows, grid, room_width, room_height, cell_size)
        place_doors(doors, grid, room_width, room_height, cell_size)
        place_furniture(furniture, grid, room_width, room_height, cell_size)

        # Draw each object on the layout
        for obj in layout_objects:
            # Skip objects without 'adjusted_position' (unplaced objects)
            if 'adjusted_position' not in obj:
                continue  # Skip unplaced objects

            obj_class = obj['class']
            real_width, real_height = obj['size']
            adjusted_x, adjusted_y = obj['adjusted_position']

            # Convert real-world coordinates to canvas pixels
            canvas_x = int(adjusted_x * layout_scaling_factor_x)
            canvas_y = int(adjusted_y * layout_scaling_factor_y)
            canvas_width = int(real_width * layout_scaling_factor_x)
            canvas_height = int(real_height * layout_scaling_factor_y)

            # Calculate top-left and bottom-right coordinates
            top_left = (int(canvas_x - canvas_width / 2), int(canvas_y - canvas_height / 2))
            bottom_right = (int(canvas_x + canvas_width / 2), int(canvas_y + canvas_height / 2))

            # Ensure coordinates are within the canvas
            top_left = (max(0, top_left[0]), max(0, top_left[1]))
            bottom_right = (min(canvas_size_pixels[0]-1, bottom_right[0]), min(canvas_size_pixels[1]-1, bottom_right[1]))

            # Draw the object
            cv2.rectangle(layout_image, top_left, bottom_right, (255, 255, 255), -1)  # White fill
            cv2.rectangle(layout_image, top_left, bottom_right, (0, 0, 0), 1)  # Black outline

            # Place label at the center
            label_x = (top_left[0] + bottom_right[0]) // 2 - 10  # Adjust for text width
            label_y = (top_left[1] + bottom_right[1]) // 2 + 5
            cv2.putText(layout_image, obj_class, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Encode the image to base64
        is_success, buffer = cv2.imencode('.png', layout_image)
        if not is_success:
            response = jsonify({'error': 'Failed to encode image'})
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response, 500

        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Return response with CORS headers
        response = jsonify({'layout_image': encoded_image})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except Exception as e:
        # Log the exception traceback
        traceback.print_exc()
        # Return error response with CORS headers
        response = jsonify({'error': 'An error occurred during processing.'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500

if __name__ == '__main__':
    app.run(debug=True)
