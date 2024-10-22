import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys

# Define fixed sizes for furniture classes (width, height in meters)
furniture_sizes = {
    'Bed': (1.5, 2.0),
    'Chair': (0.5, 0.5),
    'Door': (1.0, 0.1),
    'Sofa': (1.2, 0.6),
    'Table': (0.5, 0.3),
    'Window': (1.0, 0.1),
    # Add more classes as needed
}

# List of classes to ignore (decorative or structural elements)
ignored_classes = []  # Empty since all classes are relevant

# Mapping of image indices to wall names and orientations
wall_mapping = {
    0: {'name': 'Left', 'orientation': 'Vertical'},
    1: {'name': 'Top', 'orientation': 'Horizontal'},
    2: {'name': 'Right', 'orientation': 'Vertical'},
    3: {'name': 'Bottom', 'orientation': 'Horizontal'}
}

def load_images(image_paths):
    """Load multiple images from given paths."""
    images = []
    print("Loading images...")
    for idx, path in enumerate(image_paths):
        print(f"Loading image {idx + 1}: {path}")
        image = cv2.imread(path)
        if image is None:
            print(f"Error: Image '{path}' not found or unable to load.")
            sys.exit()
        images.append(image)
        print(f"Image '{path}' loaded successfully.\n")
    print(f"Total images loaded: {len(images)}\n")
    return images

def perform_object_detection(model, image, class_name_mapping, image_idx):
    """Perform object detection on a single image and return detected objects."""
    print(f"Performing object detection on Image {image_idx + 1}...")
    results = model(image)
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
    print(f"Detected {len(detected_objects)} relevant objects in Image {image_idx + 1}.")
    for obj in detected_objects:
        bbox = obj['bbox']
        print(f" - Class: {obj['class']}, Confidence: {obj['confidence']:.2f}, BBox: {bbox}")
    print()
    return detected_objects

def get_reference_object(detected_objects, image, image_idx):
    """Identify the furniture object closest to the image center."""
    print(f"Selecting reference object from Image {image_idx + 1}...")
    if not detected_objects:
        print("No furniture objects detected.")
        sys.exit()

    image_height, image_width = image.shape[:2]
    center_x, center_y = image_width / 2, image_height / 2
    print(f"Image {image_idx + 1} Dimensions: Width = {image_width}px, Height = {image_height}px")
    print(f"Image Center: ({center_x}px, {center_y}px)")

    # Calculate distance of each object to the center of the image
    for obj in detected_objects:
        bbox = obj['bbox']
        obj_center_x = (bbox[0] + bbox[2]) / 2
        obj_center_y = (bbox[1] + bbox[3]) / 2
        distance = np.sqrt((obj_center_x - center_x) ** 2 + (obj_center_y - center_y) ** 2)
        obj['distance_to_center'] = distance
        print(f" - Class: {obj['class']}, Distance to Center: {distance:.2f}px")

    # Select the object with the minimum distance to the center
    reference_object = min(detected_objects, key=lambda obj: obj['distance_to_center'])
    print(f"Selected Reference Object: {reference_object['class']}, BBox: {reference_object['bbox']}, Distance to Center: {reference_object['distance_to_center']:.2f}px\n")
    return reference_object

def calculate_scaling_factor(reference_bbox, reference_class, image_height_px, image_idx, distance_constant=0.026):
    """Calculate the scaling factor based on the reference object's bounding box height."""
    print(f"Calculating scaling factor for Image {image_idx + 1}...")
    reference_pixel_height = reference_bbox[3] - reference_bbox[1]
    print(f"Reference Object Pixel Height: {reference_pixel_height} pixels")

    # Compute scaling factor
    # These constants should be based on your camera's specifications
    # Adjust these values as per your actual camera parameters
    sensor_height_mm = 24  # Example value; replace with actual sensor height
    print(f"Sensor Height: {sensor_height_mm} mm")
    print(f"Image Height: {image_height_px} pixels")
    h = (reference_pixel_height * (sensor_height_mm / image_height_px)) / 1000  # Convert to meters
    print(f"Computed 'h' (meters): {h:.6f}")

    # Calculate the real distance between camera and object
    # Adjust the constant based on your setup (e.g., camera height, lens properties)
    real_distance_between_camera_and_object = distance_constant / h
    print(f"Real Distance Between Camera and Object: {real_distance_between_camera_and_object:.6f} meters\n")

    return real_distance_between_camera_and_object

def calculate_room_dimension(idx, image, model, class_name_mapping):
    """Calculate either room width or height based on the image index."""
    detected_objects = perform_object_detection(model, image, class_name_mapping, idx)

    if not detected_objects:
        print("No relevant objects detected in image index:", idx)
        sys.exit()

    reference_object = get_reference_object(detected_objects, image, idx)

    reference_class = reference_object['class']
    reference_bbox = reference_object['bbox']

    # Get image height in pixels
    image_height_px = image.shape[0]
    print(f"Image {idx + 1} Height: {image_height_px} pixels")

    # Calculate scaling factor
    scaling_factor = calculate_scaling_factor(reference_bbox, reference_class, image_height_px, idx)

    # Retrieve the fixed size of the reference object from furniture_sizes
    if reference_class in furniture_sizes:
        furniture_width, furniture_height = furniture_sizes[reference_class]
        print(f"Reference Class '{reference_class}' Fixed Size: Width = {furniture_width} m, Height = {furniture_height} m")
    else:
        print(f"Reference class '{reference_class}' not found in furniture_sizes.")
        sys.exit()

    # Calculate room dimension by adding real distance to the furniture size
    if idx == 0:
        # First image for room height (Left Wall)
        room_dimension = (furniture_height + scaling_factor)
        dimension_type = 'Height'
    elif idx == 1:
        # Second image for room width (Top Wall)
        room_dimension = (furniture_height + scaling_factor)
        dimension_type = 'Width'
    else:
        room_dimension = None
        dimension_type = None

    if room_dimension is not None:
        print(f"Calculated {dimension_type}: {room_dimension:.2f} meters (Total)\n")
    return room_dimension, reference_bbox, dimension_type

def adjust_object_positions(objects, wall_length):
    """Adjust positions of objects to avoid overlaps along a wall."""
    if not objects:
        return objects

    # Sort objects by their position along the wall
    objects.sort(key=lambda obj: obj['position'])

    for i in range(1, len(objects)):
        prev_obj = objects[i - 1]
        curr_obj = objects[i]

        # Calculate the end position of the previous object
        prev_end = prev_obj['position'] + prev_obj['size_along_wall'] / 2
        # Calculate the start position of the current object
        curr_start = curr_obj['position'] - curr_obj['size_along_wall'] / 2

        # If they overlap, adjust the current object's position
        overlap = prev_end - curr_start
        if overlap > 0:
            # Shift the current object forward to eliminate overlap
            shift = overlap + 0.01  # Adding small padding
            curr_obj['position'] += shift
            # Ensure the current object's position is within bounds
            curr_obj['position'] = min(wall_length - curr_obj['size_along_wall'] / 2, curr_obj['position'])
    return objects

def draw_room(room_width, room_height, walls_objects):
    """Draw a simple 2D representation of the room with objects on walls."""
    print("Drawing the estimated room dimensions and objects...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the room as a rectangle
    room = plt.Rectangle((0, 0), room_width, room_height, fill=None, edgecolor='black', linewidth=2)
    ax.add_patch(room)

    # Define wall positions
    walls = {
        'Top': {'start': (0, room_height), 'end': (room_width, room_height), 'orientation': 'Horizontal'},
        'Right': {'start': (room_width, 0), 'end': (room_width, room_height), 'orientation': 'Vertical'},
        'Bottom': {'start': (0, 0), 'end': (room_width, 0), 'orientation': 'Horizontal'},
        'Left': {'start': (0, 0), 'end': (0, room_height), 'orientation': 'Vertical'}
    }

    # Draw and annotate walls with objects
    for wall_name, wall_info in walls.items():
        if wall_name not in walls_objects:
            continue  # Skip walls without images

        orientation = wall_info['orientation']
        print(f"Drawing {wall_name} Wall ({orientation})...")
        # Draw wall line
        wall_line = plt.Line2D(
            [wall_info['start'][0], wall_info['end'][0]],
            [wall_info['start'][1], wall_info['end'][1]],
            color='black',
            linewidth=2
        )
        ax.add_line(wall_line)

        wall_objects = walls_objects[wall_name]['objects']
        wall_length = room_width if orientation == 'Horizontal' else room_height

        # Prepare objects: swap dimensions if necessary, set size_along_wall
        for obj in wall_objects:
            obj_class = obj['class']
            obj_width = obj['real_width']
            obj_height = obj['real_height']

            if orientation == 'Vertical':
                swapped_width = obj_height
                swapped_height = obj_width
            else:
                swapped_width = obj_width
                swapped_height = obj_height

            obj['swapped_width'] = swapped_width
            obj['swapped_height'] = swapped_height
            obj['size_along_wall'] = swapped_width  # Size along the wall

        # Adjust object positions
        # Separate furniture from windows
        furniture = [obj for obj in wall_objects if obj['class'] != 'Window']
        windows = [obj for obj in wall_objects if obj['class'] == 'Window']
        furniture = adjust_object_positions(furniture, wall_length)

        for obj in windows:
            print(f" - Detected {len(windows)} window(s) on {wall_name} Wall.")
            for idx, window in enumerate(windows):
                window_width = window['real_width']
                window_height = window['real_height']

                if orientation == 'Horizontal':
                    # Distribute windows evenly along the wall
                    spacing = room_width / (len(windows) + 0.2)
                    window_x = spacing * (idx + 1) - window_width / 2
                    window_y = wall_info['start'][1]  # Align with wall y-position
                    window_y = window_y - 0.05

                    # Draw window as a rectangle
                    window_rect = plt.Rectangle((window_x, window_y), window_width, window_height,  facecolor='white', edgecolor='black', alpha=0.7)
                    ax.add_patch(window_rect)

                    # Annotate the window
                    plt.text(window_x + window_width / 2, window_y + window_height / 2, 'Window',
                             ha='center', va='center', fontsize=8, color='black')
                else:  # Vertical
                    # Distribute windows evenly along the wall
                    spacing = room_height / (len(windows) + 0.5)
                    window_y = spacing * (idx + 1) - window_height / 2
                    window_x = wall_info['start'][0]  # Align with wall x-position

                    window_x = window_x - 0.05

                    # Draw window as a rectangle
                    window_rect = plt.Rectangle((window_x, window_y), window_height, window_width,  facecolor='white', edgecolor='black', alpha=0.7)
                    ax.add_patch(window_rect)

                    # Annotate the window
                    plt.text(window_x + window_height / 2, window_y + window_width / 2, 'Window',
                             ha='center', va='center', fontsize=8, color='black')
            print(f" - Placed {len(windows)} window(s) on {wall_name} Wall.\n")
        else:
            print(f" - No windows detected on {wall_name} Wall.\n")

        # Update positions in wall_objects for adjusted furniture
        for obj in wall_objects:
            if obj['class'] != 'Window':
                # Find the adjusted position from furniture
                adjusted_obj = next((f_obj for f_obj in furniture if f_obj is obj), None)
                if adjusted_obj:
                    obj['position'] = adjusted_obj['position']

        # Now proceed to draw the objects
        for obj in furniture:
            obj_class = obj['class']
            swapped_width = obj['swapped_width']
            swapped_height = obj['swapped_height']
            obj_position = obj['position']

            if orientation == 'Vertical':
                x_position = wall_info['start'][0]
                if wall_name == 'Left':
                    obj_x = x_position + 0.05
                else:
                    obj_x = x_position - 0.05 - swapped_width
                obj_y = obj_position - swapped_height / 2
                rect_width = swapped_width
                rect_height = swapped_height
            else:
                obj_x = obj_position - swapped_width / 2
                y_position = wall_info['start'][1]
                if wall_name == 'Top':
                    obj_y = y_position - 0.05 - swapped_height
                else:
                    obj_y = y_position + 0.05
                rect_width = swapped_width
                rect_height = swapped_height

            # Draw the object as a rectangle
            obj_rect = plt.Rectangle((obj_x, obj_y), rect_width, rect_height,
                                     facecolor='white', edgecolor='black', alpha=0.7)
            ax.add_patch(obj_rect)

            # Annotate the object
            plt.text(obj_x + rect_width / 2, obj_y + rect_height / 2, obj_class,
                     ha='center', va='center', fontsize=8, color='black')

        print(f" - Placed {len(wall_objects)} object(s) on {wall_name} Wall.\n")

    # Set limits and labels
    ax.set_xlim(-1, room_width + 1)
    ax.set_ylim(-1, room_height + 1)
    ax.set_xlabel('Meters')
    ax.set_ylabel('Meters')
    ax.set_title('Estimated Room Dimensions with Objects')
    ax.set_aspect('equal')
    plt.grid(False)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()
    print("Room dimensions and objects drawn successfully.\n")


def main(image_paths):
    # Validate number of images
    if not (2 <= len(image_paths) <= 4):
        print("Error: Please provide at least 2 and at most 4 image paths.")
        sys.exit()

    print(f"Starting processing for {len(image_paths)} images...\n")

    # Load images
    images = load_images(image_paths)

    # Initialize the YOLO model
    print("Initializing YOLO model...")
    try:
        model = YOLO('model_09.pt')  # Replace with your trained model
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        sys.exit()
    print("YOLO model initialized successfully.\n")

    # Define class name mapping (optional)
    class_name_mapping = {
        'Bed': 'Bed',
        'Chair': 'Chair',
        'Door': 'Door',
        'Sofa': 'Sofa',
        'Table': 'Table',
        'Window': 'Window',
        # Add more mappings if needed
    }

    # Define wall mapping based on image index
    wall_mapping_indices = {
        0: 'Left',
        1: 'Top',
        2: 'Right',
        3: 'Bottom'
    }

    # Calculate room dimensions from the first two images
    room_dimensions = {}
    walls_present = []

    for idx in range(2):  # Use only first two images for size
        if idx >= len(images):
            print(f"Warning: Less than {idx + 1} images provided. Skipping Image {idx + 1}.")
            continue
        wall = wall_mapping_indices.get(idx, f'Wall_{idx+1}')
        print(f"Processing Image {idx + 1} for {'Height' if idx == 0 else 'Width'} calculation...")
        dimension, bbox, dim_type = calculate_room_dimension(idx, images[idx], model, class_name_mapping)
        if dim_type:
            room_dimensions[dim_type] = dimension
            walls_present.append(wall)

    if not room_dimensions:
        print("Error: Unable to calculate room dimensions.")
        sys.exit()

    print(f"\nEstimated Room Dimensions (Total):")
    for dim, value in room_dimensions.items():
        print(f" - {dim}: {value:.2f} meters")
    print()

    # Ensure room dimensions are calculated
    room_width = room_dimensions.get('Width', None)
    room_height = room_dimensions.get('Height', None)
    if room_width is None or room_height is None:
        print("Error: Room dimensions not properly calculated.")
        sys.exit()

    # Prepare to collect objects for each wall
    walls_objects = {}

    # Process all images and collect objects for corresponding walls
    for idx, image in enumerate(images):
        if idx >= 4:
            print("Warning: More than 4 images provided. Only the first 4 will be processed.")
            break
        wall_info = wall_mapping.get(idx)
        if not wall_info:
            print(f"No mapping defined for Image {idx + 1}. Skipping...")
            continue
        wall_name = wall_info['name']
        orientation = wall_info['orientation']

        print(f"Processing Image {idx + 1} for object placement on {wall_name} Wall...")
        detected_objects = perform_object_detection(model, image, class_name_mapping, idx)

        if not detected_objects:
            print(f"No relevant objects detected in Image {idx + 1}. Skipping {wall_name} Wall.\n")
            continue

        # Get image dimensions
        image_height_px, image_width_px = image.shape[:2]
        print(f"Image {idx + 1} Dimensions: Width = {image_width_px}px, Height = {image_height_px}px")

        # Compute scaling factors based on room dimensions
        if orientation == 'Horizontal':
            meters_per_pixel = room_width / image_width_px
        else:
            meters_per_pixel = room_height / image_width_px  # Use image_width_px due to swapping

        # For each object, compute real positions
        objects_real_sizes = []
        for obj in detected_objects:
            obj_class = obj['class']
            if obj_class in furniture_sizes:
                obj_width = furniture_sizes[obj_class][0]
                obj_height = furniture_sizes[obj_class][1]
            else:
                obj_width, obj_height = 0.5, 0.5  # Default size if not found

            bbox = obj['bbox']
            obj_center_x = (bbox[0] + bbox[2]) / 2  # in pixels
            obj_center_y = (bbox[1] + bbox[3]) / 2  # in pixels

            # Convert to meters
            if orientation == 'Horizontal':
                obj_position = obj_center_x * meters_per_pixel
            else:
                obj_position = obj_center_x * meters_per_pixel  # Swap x and y

            objects_real_sizes.append({
                'class': obj_class,
                'real_width': obj_width,
                'real_height': obj_height,
                'position': obj_position,  # position along wall (in meters)
            })

        # Store objects for the wall
        walls_objects[wall_name] = {
            'orientation': orientation,
            'objects': objects_real_sizes
        }
        print(f"Collected {len(objects_real_sizes)} objects for {wall_name} Wall.\n")

    # Draw the room with objects
    draw_room(room_width, room_height, walls_objects)

if __name__ == "__main__":
    # Example usage:
    # Provide a list of image paths (minimum 2, maximum 4)
    image_paths = ['123.jpeg', '124.jpeg']  # Replace with your actual image files
    main(image_paths)
