import cv2
import numpy as np
import json
import math
import numpy as np
from sklearn.cluster import DBSCAN

def load_image_with_alpha(image_path):
    """
    Load an image ensuring it has an alpha channel.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - img: Loaded image with an alpha channel.

    Raises:
    - ValueError: If the image cannot be loaded or lacks an alpha channel.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    if img.shape[2] != 4:
        raise ValueError(f"Input image '{image_path}' does not have an alpha channel.")
    return img

def create_binary_mask_from_alpha(img):
    """
    Create a binary mask from the alpha channel of an image.

    Parameters:
    - img: Image with an alpha channel.

    Returns:
    - binary_mask: Binary mask where non-transparent regions are 255.
    """
    alpha_channel = img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    return binary_mask

def save_image_with_alpha(img, output_path):
    """
    Save an image that includes an alpha channel.

    Parameters:
    - img: Image to save.
    - output_path: Path to save the image.

    Returns:
    - None
    """
    cv2.imwrite(output_path, img)
    print(f"Image saved at {output_path}")

def expand_and_save_layer(input_image_path, output_path, expansion_amount, draw_outline=False, outline_color=(0,0,0,255), outline_thickness=2):
    """
    Expand the non-transparent regions of an image and save the result.

    Parameters:
    - input_image_path: Path to the input image.
    - output_path: Path to save the output image.
    - expansion_amount: Amount to expand (dilate) the regions.
    - draw_outline: Whether to draw an outline around the expanded regions.
    - outline_color: Color of the outline (if draw_outline is True).
    - outline_thickness: Thickness of the outline (if draw_outline is True).
    """
    img = load_image_with_alpha(input_image_path)
    binary_mask = create_binary_mask_from_alpha(img)

    # Expand (dilate) the binary mask
    kernel = np.ones((expansion_amount, expansion_amount), np.uint8)
    expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Create an empty transparent image for the expanded layer
    expanded_layer = np.zeros_like(img)
    expanded_layer[:, :, 0:3] = 255  # Set RGB to white
    expanded_layer[:, :, 3] = expanded_mask  # Set alpha to the expanded mask

    if draw_outline:
        # Find contours of the expanded regions
        contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw the outline
        cv2.drawContours(expanded_layer, contours, -1, outline_color, thickness=outline_thickness)

    save_image_with_alpha(expanded_layer, output_path)

def expand_columns(column_image_path, output_path, expansion_amount):
    """
    Expand the columns by dilating them and add a black outline.

    Parameters:
    - column_image_path: Path to the input column image.
    - output_path: Path to save the expanded columns image.
    - expansion_amount: Amount to expand (dilate) the columns.
    """
    expand_and_save_layer(column_image_path, output_path, expansion_amount, draw_outline=True)

def create_footing_layer(column_image_path, output_path, expansion_amount):
    """
    Create a footing layer by expanding the column contours without any black outline.

    Parameters:
    - column_image_path: Path to the input column image.
    - output_path: Path to save the footing layer image.
    - expansion_amount: Amount to expand (dilate) the footings.
    """
    expand_and_save_layer(column_image_path, output_path, expansion_amount, draw_outline=False)

def add_padding_to_walls(wall_image_path, output_path, padding_amount):
    """
    Add padding to line-like walls by dilating the lines without any black outline.

    Parameters:
    - wall_image_path: Path to the input wall image.
    - output_path: Path to save the padded walls image.
    - padding_amount: Amount to expand (dilate) the walls.
    """
    expand_and_save_layer(wall_image_path, output_path, padding_amount, draw_outline=False)

def create_wall_annotation_layer(json_file, output_path, conversion_factor=1, text_color=(255,0,0,255)):
    """
    Create an annotation layer for walls based on the JSON file, displaying wall lengths.

    Parameters:
    - json_file: Path to the JSON file containing wall data.
    - output_path: Path to save the annotation layer.
    - conversion_factor: Conversion factor for units.
    - text_color: Color of the text annotations.
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get canvas dimensions from JSON features
    canvas_width = data['features']['canvas_width']
    canvas_height = data['features']['canvas_height']

    # Create an empty transparent image for the annotation layer
    annotation_layer = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Add wall length annotations
    for wall in data['walls']:
        start_x, start_y = wall['start_point']
        end_x, end_y = wall['end_point']
        length_cm = wall['length_cm']  # Length in cm from JSON

        # Calculate the midpoint of the wall
        mid_x = int((start_x + end_x) / 2)
        mid_y = int((start_y + end_y) / 2)

        # Convert length to display units (using the conversion factor)
        length_display = round(length_cm * conversion_factor, 2)

        # Add text annotation to the midpoint
        cv2.putText(annotation_layer, f"{length_display} mm", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save the annotation layer as a PNG with transparency
    save_image_with_alpha(annotation_layer, output_path)

def create_annotation_layer_from_image(image_path, output_path, conversion_factor=1, offset=10, text_color=(0,255,0,255), annotate_only_one=False):
    """
    Create an annotation layer by calculating dimensions from an image.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path to save the annotation layer.
    - conversion_factor: Conversion factor from pixels to display units.
    - offset: Vertical offset for the text annotations.
    - text_color: Color of the text annotations.
    - annotate_only_one: If True, only annotate one contour.
    """
    img = load_image_with_alpha(image_path)
    binary_mask = create_binary_mask_from_alpha(img)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty transparent image for the annotation layer
    annotation_layer = np.zeros_like(img)

    # If annotate_only_one is True, select one contour (e.g., the largest)
    if annotate_only_one and contours:
        contours = [max(contours, key=cv2.contourArea)]

    # Iterate through each contour and calculate its dimensions
    for contour in contours:
        # Calculate the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate dimension using the conversion factor
        dimension = round(w * conversion_factor, 2)

        # Calculate the position for the annotation (slightly above the bounding box)
        text_x = int(x + w / 2)
        text_y = max(0, y - offset)  # Ensure text doesn't go out of bounds

        # Add text annotation
        cv2.putText(annotation_layer, f"{dimension} cm", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # If only annotating one contour, break after the first
        if annotate_only_one:
            break

    # Save the annotation layer
    save_image_with_alpha(annotation_layer, output_path)
    
def create_manual_footing_annotation_layer(expanded_column_image, conversion_factor=1, offset=10):
    """
    Create an annotation layer for footings by manually calculating their dimensions from the expanded column image.

    Parameters:
    - expanded_column_image: Path to the input expanded column image.
    - conversion_factor: Conversion factor for units.
    - offset: Vertical offset for text annotations.

    Returns:
    - annotation_layer: NumPy array representing the annotation layer.
    - footing_size_cm: Size of the footing in cm, calculated from the largest contour.
    """
    img = load_image_with_alpha(expanded_column_image)
    binary_mask = create_binary_mask_from_alpha(img)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty transparent image for the annotation layer
    annotation_layer = np.zeros_like(img)

    footing_size_cm = 0  # Initialize footing size variable

    # If there are contours, select the largest (as 'footing')
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate dimension in cm using the conversion factor
        footing_size_cm = round(w * conversion_factor, 2)
        footing_size_m = footing_size_cm/1000
        # Calculate the position for the annotation (slightly above the bounding box)
        text_x = int(x + w / 2)
        text_y = max(0, y - offset)

        # Add text annotation to the layer
        cv2.putText(annotation_layer, f"{footing_size_m*1000} mm", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1, cv2.LINE_AA)

    return annotation_layer, footing_size_cm


def create_expanded_column_annotation_layer(expanded_column_image, output_path, conversion_factor=1, offset=10):
    """
    Create an annotation layer for the expanded columns by calculating their dimensions.
    """
    create_annotation_layer_from_image(
        expanded_column_image,
        output_path,
        conversion_factor,
        offset,
        text_color=(0,255,0,255),
        annotate_only_one=True  # Only annotate one column
    )
def draw_dashed_line(img, start_point, end_point, color=(0, 0, 0, 255), thickness=1, dash_length=20, gap_length=5):
    """
    Draw a dashed line on an image from start_point to end_point.

    Parameters:
    - img: The image on which to draw.
    - start_point: Starting point of the line (x, y).
    - end_point: Ending point of the line (x, y).
    - color: Color of the line.
    - thickness: Thickness of the line.
    - dash_length: Length of each dash.
    - gap_length: Length of each gap.
    """
    # Calculate the Euclidean distance between the start and end points
    total_length = np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
    # Calculate the number of dash-gaps
    num_dashes = int(total_length // (dash_length + gap_length))

    # Calculate the direction vector
    direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    direction = (direction[0] / total_length, direction[1] / total_length)

    for i in range(num_dashes + 1):
        # Start point of the dash
        start = (
            int(start_point[0] + (dash_length + gap_length) * i * direction[0]),
            int(start_point[1] + (dash_length + gap_length) * i * direction[1])
        )
        # End point of the dash
        end = (
            int(start[0] + dash_length * direction[0]),
            int(start[1] + dash_length * direction[1])
        )
        # Ensure the end point does not exceed the total length
        if np.hypot(end[0] - start_point[0], end[1] - start_point[1]) > total_length:
            end = end_point
        # Draw the dash
        cv2.line(img, start, end, color, thickness)

def create_dashed_grid_cross_layer(json_file, output_path, canvas_width, canvas_height, eps=50):
    """
    Create a grid-like cross marker layer with black dashed lines based on the column center points from the JSON file.
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create an empty transparent image for the grid cross marker layer
    grid_cross_layer = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Collect all center points
    centers = []
    for column in data['columns']:
        start_x, start_y = column['start_point']
        end_x, end_y = column['end_point']
        center_x = int((start_x + end_x) / 2)
        center_y = int((start_y + end_y) / 2)
        centers.append((center_x, center_y))

    centers_np = np.array(centers)

    # Debug: Print column centers
    print("Column Centers:")
    for idx, (cx, cy) in enumerate(centers):
        print(f"Column {idx}: ({cx}, {cy})")

    # Cluster on x-coordinates
    x_coords = centers_np[:, 0].reshape(-1, 1)
    clustering_x = DBSCAN(eps=eps, min_samples=1).fit(x_coords)
    labels_x = clustering_x.labels_

    # Cluster on y-coordinates
    y_coords = centers_np[:, 1].reshape(-1, 1)
    clustering_y = DBSCAN(eps=eps, min_samples=1).fit(y_coords)
    labels_y = clustering_y.labels_

    # Debug: Print cluster labels
    print("X-coordinate cluster labels:", labels_x)
    print("Y-coordinate cluster labels:", labels_y)

    # For each cluster in x, calculate the average x position
    unique_labels_x = np.unique(labels_x)
    averaged_centers_x = []
    for label in unique_labels_x:
        cluster_points = x_coords[labels_x == label]
        avg_x = int(np.mean(cluster_points))
        averaged_centers_x.append(avg_x)
        print(f"Cluster X {label}: Avg X = {avg_x}")

    # For each cluster in y, calculate the average y position
    unique_labels_y = np.unique(labels_y)
    averaged_centers_y = []
    for label in unique_labels_y:
        cluster_points = y_coords[labels_y == label]
        avg_y = int(np.mean(cluster_points))
        averaged_centers_y.append(avg_y)
        print(f"Cluster Y {label}: Avg Y = {avg_y}")

    # Draw vertical dashed lines at averaged x positions (extend to match canvas height)
    for avg_x in averaged_centers_x:
        draw_dashed_line(grid_cross_layer, (avg_x, 0), (avg_x, canvas_height), color=(0, 0, 0, 255))

    # Draw horizontal dashed lines at averaged y positions (extend to match canvas width)
    for avg_y in averaged_centers_y:
        draw_dashed_line(grid_cross_layer, (0, avg_y), (canvas_width, avg_y), color=(0, 0, 0, 255))

    # Save the grid cross marker layer with dashed lines as a PNG with transparency
    save_image_with_alpha(grid_cross_layer, output_path)
    print(f"Dashed grid cross marker layer saved at {output_path}")
  
def draw_dashed_outline(img, contours, color=(0, 0, 0, 255), thickness=1, dash_length=20, gap_length=5):
    """
    Draw a dashed outline around the specified contours.

    Parameters:
    - img: The image on which to draw.
    - contours: List of contours (from cv2.findContours).
    - color: Color of the dashed outline.
    - thickness: Thickness of the outline.
    - dash_length: Length of each dash.
    - gap_length: Length of each gap.
    """
    for contour in contours:
        points = contour.squeeze()
        if len(points.shape) == 1:
            continue  # Not enough points to draw

        num_points = len(points)
        for i in range(num_points):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % num_points])  # Wrap around to the first point
            draw_dashed_line(img, pt1, pt2, color, thickness, dash_length, gap_length)

def combine_layers_and_add_dashed_outline(footing_path, padded_walls_path, output_path):
    """
    Combine the padded walls and footing layers, then add a dashed outline for both outer and inner contours.

    Parameters:
    - footing_path: Path to the footing layer image.
    - padded_walls_path: Path to the padded walls image.
    - output_path: Path to save the combined image.
    """
    # Load the footing and padded wall images
    footing_img = load_image_with_alpha(footing_path)
    padded_walls_img = load_image_with_alpha(padded_walls_path)

    # Ensure the images are the same size
    if footing_img.shape != padded_walls_img.shape:
        raise ValueError("Footing and padded wall images must have the same dimensions.")

    # Combine the images by taking the maximum alpha channel values
    combined_layer = np.zeros_like(footing_img)
    combined_layer[:, :, 0:3] = 255  # Set RGB to white
    combined_layer[:, :, 3] = np.maximum(footing_img[:, :, 3], padded_walls_img[:, :, 3])

    # Find contours of the combined mask (using cv2.RETR_TREE to detect both outer and inner contours)
    _, combined_mask = cv2.threshold(combined_layer[:, :, 3], 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a dashed outline around all contours (both outer and inner)
    draw_dashed_outline(combined_layer, contours, color=(0, 0, 0, 255), thickness=1)

    # Save the combined layer with dashed outline as a PNG with transparency
    save_image_with_alpha(combined_layer, output_path)

def overlay_images(base_image, overlay_image):
    """
    Overlay one image on top of another, handling transparency.

    Parameters:
    - base_image: The base image.
    - overlay_image: The image to overlay on top.

    Returns:
    - None
    """
    # Ensure the images are the same size
    if base_image.shape != overlay_image.shape:
        raise ValueError("Base image and overlay image must have the same dimensions.")

    # Blend the images based on alpha channel
    alpha_overlay = overlay_image[:, :, 3] / 255.0
    for c in range(3):  # Iterate over RGB channels
        base_image[:, :, c] = (1 - alpha_overlay) * base_image[:, :, c] + alpha_overlay * overlay_image[:, :, c]

    # Update the alpha channel
    base_image[:, :, 3] = np.maximum(base_image[:, :, 3], overlay_image[:, :, 3])

def combine_all_layers(layers, output_path, canvas_size, grid_layer_path=None):
    """
    Combine all specified layers from bottom to top with a white background.

    Parameters:
    - layers: List of layer image paths in order from topmost to bottommost.
    - output_path: Path to save the final combined image.
    - canvas_size: Tuple of (width, height) for the canvas size.
    - grid_layer_path: Path to the grid layer to overlay at the center without scaling.
    """
    width, height = canvas_size

    # Create a white background image with transparency
    final_image = np.zeros((height, width, 4), dtype=np.uint8)
    final_image[:, :, 0:3] = 255  # Set RGB to white
    final_image[:, :, 3] = 255  # Set alpha to fully opaque

    # Overlay each layer in reverse order (bottom to top)
    for layer_path in reversed(layers):
        layer_img = load_image_with_alpha(layer_path)

        if layer_path == grid_layer_path:
            # Overlay the grid layer at the center without scaling
            final_image = overlay_images_at_center(final_image, layer_img)
        else:
            # Resize other layers to match the canvas size
            if layer_img.shape[:2] != (height, width):
                layer_img = cv2.resize(layer_img, (width, height))

            # Overlay the layer on the final image
            overlay_images(final_image, layer_img)

    # Save the final combined image as a PNG
    save_image_with_alpha(final_image, output_path)

  
    
def add_padding_to_json(json_data, padding_percentage=0.20):
    """
    Adjusts the coordinates in the JSON data to account for added padding.

    Parameters:
    - json_data: The JSON data to modify.
    - padding_percentage: The percentage of padding to add around the canvas.

    Returns:
    - json_data: The modified JSON data with updated coordinates.
    """
    # Get the original canvas dimensions
    original_width = json_data['features']['canvas_width']
    original_height = json_data['features']['canvas_height']

    # Calculate the padding amounts
    pad_w = int(original_width * padding_percentage)
    pad_h = int(original_height * padding_percentage)

    # Update the canvas dimensions to include the padding
    new_width = original_width + 2 * pad_w
    new_height = original_height + 2 * pad_h
    json_data['features']['canvas_width'] = new_width
    json_data['features']['canvas_height'] = new_height

    # Adjust the coordinates for columns
    for column in json_data['columns']:
        column['start_point'][0] += pad_w
        column['start_point'][1] += pad_h
        column['end_point'][0] += pad_w
        column['end_point'][1] += pad_h

    # Adjust the coordinates for walls
    for wall in json_data['walls']:
        wall['start_point'][0] += pad_w
        wall['start_point'][1] += pad_h
        wall['end_point'][0] += pad_w
        wall['end_point'][1] += pad_h

    return json_data


def compute_reinforcement_bars():
    """
    Placeholder function to compute the number of reinforcement bars.
    Returns a fixed number for now.
    """
    return 6  # Replace with actual computation logic when available

def calculate_square_reinforcement(footing_size, footing_thickness, bar_diameter, concrete_cover):
    # Calculate gross area of footing (Ag)
    Ag = footing_size * footing_thickness

    # Calculate minimum area of steel reinforcement (As)
    As = 0.002 * Ag

    # Calculate the number of steel bars (rounding up)
    bar_area = (math.pi / 4) * (bar_diameter ** 2)
    no_of_bars = math.ceil(As / bar_area)

    # Calculate spacing for bars
    spacing = (footing_size - 2 * concrete_cover) / no_of_bars

    return {"no_of_bars": no_of_bars, "spacing": spacing}

import random

def create_footing_info_layer(image_path, output_path, footing_size_cm, reinforcement_diameter, number_of_storeys,deadLoad,wallLoad,floorLoad,roofLoad,liveLoad,windLoad,seismicLoad,totalLoad,lengthRLTimer,lengthVAE,conversion_factor=1, offset=10):
    """
    Create an annotation layer for footing information, including reinforcement details.

    Parameters:
    - image_path: Path to the input footing image.
    - output_path: Path to save the footing info layer.
    - footing_size_cm: Size of the footing in cm.
    - reinforcement_diameter: Diameter of reinforcement bars in mm.
    - number_of_storeys: Number of storeys (1 or 2).
    - conversion_factor: Conversion factor for units (default 1).
    - offset: Vertical offset for text annotations (default 10).
    """
    # Load the image
    img = load_image_with_alpha(image_path)
    binary_mask = create_binary_mask_from_alpha(img)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a fully transparent annotation layer
    annotation_layer = np.zeros_like(img, dtype=np.uint8)

    if contours:
        # Find the footing closest to the bottom-left corner
        image_height, image_width = binary_mask.shape
        bottom_left_point = (0, image_height)
        min_distance = None
        closest_footing_contour = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            footing_center_x = x + w // 2
            footing_center_y = y + h // 2

            # Calculate distance from footing center to bottom-left corner
            distance = ((footing_center_x - bottom_left_point[0]) ** 2 + (footing_center_y - bottom_left_point[1]) ** 2) ** 0.5

            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_footing_contour = contour
                closest_footing_bbox = (x, y, w, h)

        # Use the closest footing
        x, y, w, h = closest_footing_bbox

        # Calculate the number of reinforcement bars and spacing
        reinforcement_info = calculate_square_reinforcement(footing_size_cm, 225, reinforcement_diameter, 75)
        number_of_bars = reinforcement_info['no_of_bars']
        spacing = reinforcement_info['spacing']

        # Determine depth of footing based on the number of storeys
        depth_of_footing = 1125 if number_of_storeys == 1 else 1425


        text_lines = [
            "Concrete Cover: 75 mm",
            "Footing Thickness: 225 mm",
            f"Reinforcement: {number_of_bars} pcs Desformed Steel Bar - {reinforcement_diameter} mm diameter - Spacing: {spacing:.1f} mm",
            f"Depth of Footing: {depth_of_footing} mm ({number_of_storeys} storey{'s' if number_of_storeys > 1 else ''})",
            f"Dead Load: {deadLoad:.2f} kN = {wallLoad:.2f} kN + {floorLoad:.2f} kN + {roofLoad:.2f} kN ",
            f"Total Live Load: {liveLoad:.2f} kN ",
            f"Average Wind Load: {windLoad:.2f} kN",
            f"Seismic Load: {seismicLoad:.2f} kN",
            f"Total Building Load: {totalLoad:.2f} kN",
            f"Time Taken (Algorithm/Generation) : {lengthRLTimer:.2f} seconds / {lengthVAE:.2f} seconds"
        ]
        
        # TODO: Text Lines

        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        font_thickness = 1
        text_color_bgr = (0, 0, 0)  # Black text
        box_color_bgr = (255, 255, 255, 255)  # White background with full alpha
        box_border_color = (0, 0, 0, 255)  # Black border with full alpha

        # Calculate size of text block
        text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in text_lines]
        max_text_width = max([size[0] for size in text_sizes])
        total_text_height = sum([size[1] + offset for size in text_sizes]) - offset

        # Position of text box at bottom-left corner
        box_x = offset
        box_y = annotation_layer.shape[0] - total_text_height - 2 * offset

        # Draw the text box (filled rectangle with border)
        box_width = max_text_width + 2 * offset
        box_height = total_text_height + 2 * offset
        cv2.rectangle(annotation_layer, (box_x, box_y), (box_x + box_width, box_y + box_height), box_color_bgr, cv2.FILLED)
        cv2.rectangle(annotation_layer, (box_x, box_y), (box_x + box_width, box_y + box_height), box_border_color, 1)

        # Write text lines inside the box
        current_y = box_y + offset
        for line, size in zip(text_lines, text_sizes):
            text_x = box_x + offset
            text_y = current_y + size[1]
            cv2.putText(annotation_layer, line, (text_x, text_y), font, font_scale, text_color_bgr, font_thickness, cv2.LINE_AA)
            current_y += size[1] + offset

        # Choose a random point within the expanded footing bounding box
        random_x = random.randint(x, x + w)
        random_y = random.randint(y, y + h)
        random_point_inside_footing = (random_x, random_y)

        # Draw arrow from the top of the box to the random point inside the footing
        arrow_start_point = (box_x + box_width // 2, box_y)
        arrow_end_point = random_point_inside_footing

        arrow_color = (1, 1, 1, 255)  # Black color for the arrow
        arrow_thickness = 2  # Reduced thickness
        arrow_tip_length = 0.05  # Smaller tip length

        cv2.arrowedLine(annotation_layer, arrow_start_point, arrow_end_point, arrow_color, thickness=arrow_thickness, tipLength=arrow_tip_length)

    else:
        print("No footings found in the image.")

    # Ensure alpha channel is set correctly
    b_channel, g_channel, r_channel = cv2.split(annotation_layer[:, :, :3])
    alpha_mask = ((b_channel != 0) | (g_channel != 0) | (r_channel != 0)).astype(np.uint8) * 255
    annotation_layer[:, :, 3] = alpha_mask

    # Save the annotation layer
    save_image_with_alpha(annotation_layer, output_path)
    print(f"Footing info layer saved at {output_path}")


def overlay_images_at_center(base_image, overlay_image):
    """
    Overlay one image on top of another at the center of the base image, without resizing, handling transparency.

    Parameters:
    - base_image: The base image.
    - overlay_image: The image to overlay at the center.

    Returns:
    - base_image: The modified base image with the overlay.
    """
    base_h, base_w = base_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]

    # Calculate the top-left corner for overlay placement
    x_offset = (base_w - overlay_w) // 2
    y_offset = (base_h - overlay_h) // 2

    # Ensure overlay fits within the base image dimensions
    if x_offset < 0 or y_offset < 0:
        raise ValueError("Overlay image is larger than the base image.")

    # Iterate over RGB channels and alpha channel to blend the overlay image onto the base image
    for c in range(3):  # For each RGB channel
        base_image[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w, c] = (
            overlay_image[:, :, c] * (overlay_image[:, :, 3] / 255.0)
            + base_image[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w, c] * (1 - (overlay_image[:, :, 3] / 255.0))
        )

    # Update the alpha channel
    base_image[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w, 3] = np.maximum(
        base_image[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w, 3], overlay_image[:, :, 3]
    )

    return base_image

def save_image_with_alpha(img, output_path):
    """
    Save an image that includes an alpha channel.

    Parameters:
    - img: Image to save (numpy array with 4 channels).
    - output_path: Path to save the image.

    Returns:
    - None
    """
    cv2.imwrite(output_path, img)
    print(f"Image saved at {output_path}")


# Main script to process the images
# if __name__ == "__main__":
#   # Input and output file paths
#     column_image = 'output/vae/generated_columns.png'
#     expanded_columns_output = 'output/vae/expanded_columns.png'
#     footing_output = 'output/vae/footing_layer_no_outline.png'
#     wall_image_path = 'output/vae/generated_walls.png'
#     padded_walls_output = 'output/vae/padded_walls_no_outline.png'
#     json_file_path = 'output/RL/RLFoundation.json'
#     padded_json_output = 'output/RL/RLFoundationPadded.json'
#     wall_annotation_output = 'output/vae/wall_annotation_layer.png'
#     footing_annotation_output = 'output/vae/footing_info_layer.png'
#     column_annotation_output = 'output/vae/manual_column_annotation_layer.png'
#     grid_cross_output = 'output/vae/black_dashed_grid_cross_marker_layer.png'
#     combined_layer_output = 'output/vae/combined_layer_with_dashed_outline.png'
#     footing_info_output = 'output/vae/footing_info_layer.png'  # New output file
#     final_combined_output = 'static/images/final_combined_image.png'

#     # Parameters
#     expansion_amount_columns = int(10 * column_scale)
#     expansion_amount_footing = int(90  * footing_scale)
#     padding_amount_walls = 40
#     conversion_factor = 7.46  # Example conversion factor (cm/px)
#     offset = 10  # Offset for annotations
#     padding_percentage = 0.2  # Padding percentage for JSON data

#     # Step 1: Expand the columns
#     expand_columns(column_image, expanded_columns_output, expansion_amount_columns)

#     # Step 2: Create the footing layer by expanding the columns
#     create_footing_layer(column_image, footing_output, expansion_amount_footing)

#     # Step 3: Load and modify JSON data with padding
#     with open(json_file_path, 'r') as f:
#         json_data = json.load(f)
#     padded_json_data = add_padding_to_json(json_data, padding_percentage)
#     with open(padded_json_output, 'w') as f:
#         json.dump(padded_json_data, f, indent=4)
#     print(f"Updated JSON saved at {padded_json_output}")

#     # Step 4: Add padding to the walls
#     add_padding_to_walls(wall_image_path, padded_walls_output, padding_amount_walls)

#     # Step 5: Create wall annotation layer
#     create_wall_annotation_layer(padded_json_output, wall_annotation_output, conversion_factor=1)

#     # Step 6: Create footing annotation layer and get footing size
#     footing_annotation_layer, footing_size_cm = create_manual_footing_annotation_layer(footing_output, conversion_factor, offset)

#     # Step 7: Create expanded column annotation layer
#     create_expanded_column_annotation_layer(expanded_columns_output, column_annotation_output, conversion_factor, offset)

#     # Step 8: Create dashed grid cross marker layer
#     canvas_width = padded_json_data['features']['canvas_width']
#     canvas_height = padded_json_data['features']['canvas_height']
#     create_dashed_grid_cross_layer(padded_json_output, grid_cross_output, canvas_width, canvas_height)

#     # Step 9: Combine layers and add dashed outline
#     combine_layers_and_add_dashed_outline(footing_output, padded_walls_output, combined_layer_output)

#     # Step 10: Create footing information layer
#     reinforcement_diameter = barsize_value  # User input (in mm)
#     number_of_storeys = num_storey_value  # User input (1 or 2)
#     create_footing_info_layer(footing_output, footing_info_output, footing_size_cm, reinforcement_diameter, number_of_storeys, conversion_factor, offset)
#     # Step 11: Combine all layers into the final image
#     layers = [
#         wall_annotation_output,
#         footing_annotation_output,
#         column_annotation_output,
#         footing_info_output,          # New layer with footing info
#         grid_cross_output,
#         expanded_columns_output,
#         combined_layer_output  # Bottommost layer
#     ]
#     canvas_size = (canvas_width, canvas_height)
#     combine_all_layers(layers, final_combined_output, canvas_size)