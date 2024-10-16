import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# Load the trained YOLOv8 object detection model
model = YOLO('models/yolov8/train13/weights/best.pt')  # Replace with your actual trained model path

# Conversion factor: 6.923 cm per pixel (provided by you)
px_to_cm = 6.923

# Increase the radius for corner and midpoint detection
corner_detection_radius = 150  # Increased radius for corner detection
midpoint_detection_radius = 150  # Increased radius for midpoint detection

# Function to average nearby corner points
def average_corners(corner_list, radius=corner_detection_radius):
    averaged_corners = []
    used = [False] * len(corner_list)  # Track used corners

    for i, corner in enumerate(corner_list):
        if used[i]:
            continue

        close_corners = [corner]
        for j, other_corner in enumerate(corner_list):
            if i != j and not used[j]:
                dist = np.linalg.norm(np.array(corner) - np.array(other_corner))
                if dist < radius:  # Check if corners are close enough
                    close_corners.append(other_corner)
                    used[j] = True

        # Average the close corners
        avg_corner = np.mean(close_corners, axis=0).astype(int)
        averaged_corners.append((int(avg_corner[0]), int(avg_corner[1])))
        used[i] = True

    return averaged_corners

# Function to average nearby midpoints
def average_midpoints(midpoint_list, radius=midpoint_detection_radius):
    averaged_midpoints = []
    used = [False] * len(midpoint_list)  # Track used midpoints

    for i, midpoint in enumerate(midpoint_list):
        if used[i]:
            continue

        close_midpoints = [midpoint]
        for j, other_midpoint in enumerate(midpoint_list):
            if i != j and not used[j]:
                dist = np.linalg.norm(np.array(midpoint) - np.array(other_midpoint))
                if dist < radius:  # Check if midpoints are close enough
                    close_midpoints.append(other_midpoint)
                    used[j] = True

        # Average the close midpoints
        avg_midpoint = np.mean(close_midpoints, axis=0).astype(int)
        averaged_midpoints.append((int(avg_midpoint[0]), int(avg_midpoint[1])))
        used[i] = True

    return averaged_midpoints

def generate_mask_measure_walls_annotate(image_path, json_output_path):
    # Perform inference on an image with confidence and IOU thresholds
    results = model.predict(
        source=image_path,  # Path to the image
        conf=0.02,  # Confidence threshold (2%)
        iou=1.0  # IOU threshold (100% overlap)
    )

    # Load the original image to get its dimensions
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    # Create a blank mask (black background) with the same dimensions as the original image
    wall_mask = np.zeros((height, width), dtype=np.uint8)

    # Iterate through the results and apply the bounding boxes to create the black and white mask
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get the bounding boxes (x1, y1, x2, y2 format)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])  # Convert the box coordinates to integers
            # Fill the area inside the bounding box with white (255) on the black background
            wall_mask[y1:y2, x1:x2] = 255

    # Apply morphological closing to smooth the mask and remove small gaps
    kernel = np.ones((5, 5), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)

    # Save the final black and white mask (white represents the walls)
    output_path = "output/yolov8/image/wall_mask.png"
    cv2.imwrite(output_path, wall_mask)
    print(f"Saved wall mask to {output_path}")

    # Apply edge detection to find the edges of the white walls
    edges = cv2.Canny(wall_mask, threshold1=50, threshold2=150)

    # Find contours using cv2.RETR_TREE to retrieve both outer and inner contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare JSON structure, including file name
    walls_data = {
        "file_name": os.path.basename(image_path),
        "walls": [],
        "averaged_corners": [],
        "averaged_midpoints": [],
        "features": {
            "canvas_width": width,
            "canvas_height": height
        }
    }

    all_corners = []  # Collect all corners before averaging
    all_midpoints = []  # Collect all midpoints before averaging

    # Measure the length of each edge (wall) and convert to centimeters
    for contour in contours:
        # Filter small contours (bumps) by area (adjust the threshold as needed)
        if cv2.contourArea(contour) < 100:  # Filter out small bumps
            continue

        # Approximate the contour with a higher epsilon to remove small bumps
        approx = cv2.approxPolyDP(contour, epsilon=10, closed=True)  # Increased epsilon for smoother contours

        # Measure each line segment in pixels and convert to cm
        for i in range(len(approx)):
            point1 = approx[i][0]
            point2 = approx[(i + 1) % len(approx)][0]
            length_in_pixels = np.linalg.norm(point1 - point2)  # Euclidean distance between two points
            length_in_cm = length_in_pixels * px_to_cm  # Convert to centimeters

            # Skip if the wall length is less than 900 cm
            if length_in_cm < 900:
                continue

            # Store wall data in the JSON format
            wall_data = {
                "start_point": [int(point1[0]), int(point1[1])],
                "end_point": [int(point2[0]), int(point2[1])],
                "length_cm": round(float(length_in_cm), 2)
            }
            walls_data["walls"].append(wall_data)

            # Collect corners to average later
            all_corners.append((int(point1[0]), int(point1[1])))
            all_corners.append((int(point2[0]), int(point2[1])))

            # Calculate the midpoint and collect it for averaging
            midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
            all_midpoints.append((int(midpoint[0]), int(midpoint[1])))

            # Draw the wall edge in green
            cv2.line(original_image, tuple(point1), tuple(point2), (0, 255, 0), 2)

    # Average the corners and midpoints
    averaged_corners = average_corners(all_corners, radius=corner_detection_radius)
    averaged_midpoints = average_midpoints(all_midpoints, radius=midpoint_detection_radius)

    # Add averaged corners and midpoints to the JSON
    walls_data["averaged_corners"] = [{"x": int(corner[0]), "y": int(corner[1])} for corner in averaged_corners]
    walls_data["averaged_midpoints"] = [{"x": int(midpoint[0]), "y": int(midpoint[1])} for midpoint in averaged_midpoints]

    # Save the annotated image with wall lengths, midpoints, and averaged corners
    annotated_image_path = "output/yolov8/image/annotated_image_with_averaged_corners_and_midpoints.png"
    cv2.imwrite(annotated_image_path, original_image)
    print(f"Saved annotated image with averaged corners and midpoints to {annotated_image_path}")

    # Export the wall data to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(walls_data, json_file, indent=4)
    print(f"Exported wall data to {json_output_path}")

# # Example usage
# image_path = "uploads/floorplan.jpg"  # Path to the input image
# json_output_path = 'output/yolov8/output.json'  # Path to the output JSON file

# # Generate the mask, measure the walls, and export to JSON
# generate_mask_measure_walls_annotate(image_path, json_output_path)
