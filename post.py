import cv2
import numpy as np
import json


def expand_columns(column_image, output_path, expansion_amount):
    """
    Expand the columns by dilating them and add a black outline.
    """
    # Load the original column image with transparency
    column_img = cv2.imread(column_image, cv2.IMREAD_UNCHANGED)

    if column_img is None:
        raise ValueError(f"Could not load image at {column_image}")

    # Create a binary mask from the alpha channel where columns are detected
    alpha_channel = column_img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Expand (dilate) the binary mask to add padding to the columns
    kernel = np.ones((expansion_amount, expansion_amount), np.uint8)
    expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Create an empty transparent image for the expanded column layer
    expanded_column_layer = np.zeros((column_img.shape[0], column_img.shape[1], 4), dtype=np.uint8)

    # Fill the expanded regions with white and set alpha to the expanded mask
    expanded_column_layer[:, :, 0:3] = 255  # Set RGB to white
    expanded_column_layer[:, :, 3] = expanded_mask  # Set alpha to the expanded mask

    # Find contours of the expanded regions
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the black outline around the expanded regions
    cv2.drawContours(expanded_column_layer, contours, -1, (0, 0, 0, 255), thickness=2)

    # Save the expanded column layer with a black outline as a PNG with transparency
    cv2.imwrite(output_path, expanded_column_layer)
    print(f"Expanded column layer with black outline saved at {output_path}")
       
def create_footing_layer(column_image_path, output_path, expansion_amount):
    """
    Create a footing layer by expanding the column contours without any black outline.
    """
    # Load the column image with transparency
    column_img = cv2.imread(column_image_path, cv2.IMREAD_UNCHANGED)

    if column_img is None:
        raise ValueError(f"Could not load image at {column_image_path}")

    # Check if the image has an alpha channel
    if column_img.shape[2] != 4:
        raise ValueError("Input image does not have an alpha channel.")

    # Extract the alpha channel to create a mask
    alpha_channel = column_img[:, :, 3]

    # Create a binary mask from the alpha channel where columns are detected (non-transparent regions)
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Expand (dilate) the binary mask to create the footing
    kernel = np.ones((expansion_amount, expansion_amount), np.uint8)
    expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Create an empty transparent image for the footing layer
    footing_layer = np.zeros((column_img.shape[0], column_img.shape[1], 4), dtype=np.uint8)

    # Fill the expanded contours with white (R=255, G=255, B=255, A=255)
    footing_layer[:, :, 0:3] = 255  # Set RGB to white
    footing_layer[:, :, 3] = expanded_mask  # Set alpha to the expanded mask

    # Save the footing layer as a PNG with transparency
    cv2.imwrite(output_path, footing_layer)
    print(f"Footing layer saved at {output_path}")
       
    
def add_padding_to_walls(wall_image_path, output_path, padding_amount):
    """
    Add padding to line-like walls by dilating the lines without any black outline.
    """
    # Load the wall image with transparency
    wall_img = cv2.imread(wall_image_path, cv2.IMREAD_UNCHANGED)

    if wall_img is None:
        raise ValueError(f"Could not load image at {wall_image_path}")

    # Check if the image has an alpha channel
    if wall_img.shape[2] != 4:
        raise ValueError("Input image does not have an alpha channel.")

    # Extract the alpha channel to create a binary mask
    alpha_channel = wall_img[:, :, 3]

    # Create a binary mask from the alpha channel where walls are detected (non-transparent regions)
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Expand (dilate) the binary mask to add padding to the lines
    kernel = np.ones((padding_amount, padding_amount), np.uint8)
    padded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Create an empty transparent image for the padded wall layer
    padded_wall_layer = np.zeros((wall_img.shape[0], wall_img.shape[1], 4), dtype=np.uint8)

    # Fill the padded regions with white
    padded_wall_layer[:, :, 0:3] = 255  # Set RGB to white
    padded_wall_layer[:, :, 3] = padded_mask  # Set alpha to the padded mask

    # Save the padded wall layer as a PNG with transparency
    cv2.imwrite(output_path, padded_wall_layer)
    print(f"Padded wall layer saved at {output_path}")


def create_wall_annotation_layer(json_file, output_path, conversion_factor=1):
    """
    Create an annotation layer for walls based on the JSON file, displaying wall lengths.
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
        cv2.putText(annotation_layer, f"{length_display} cm", (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 1, cv2.LINE_AA)

    # Save the annotation layer as a PNG with transparency
    cv2.imwrite(output_path, annotation_layer)
    print(f"Wall annotation layer saved at {output_path}")

def create_manual_footing_annotation_layer(expanded_column_image, output_path, conversion_factor=1, offset=10):
    """
    Create an annotation layer for footings by manually calculating their dimensions from the expanded column image.
    """
    # Load the expanded column image (footing layer) with transparency
    column_img = cv2.imread(expanded_column_image, cv2.IMREAD_UNCHANGED)

    if column_img is None:
        raise ValueError(f"Could not load image at {expanded_column_image}")

    # Create a binary mask from the alpha channel where columns are detected
    alpha_channel = column_img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the expanded columns (footings)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty transparent image for the annotation layer
    annotation_layer = np.zeros((column_img.shape[0], column_img.shape[1], 4), dtype=np.uint8)

    # Iterate through each footing and calculate its dimensions
    for contour in contours:
        # Calculate the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate dimension in cm using the conversion factor
        dimension_cm = round(w * conversion_factor, 2)

        # Calculate the position for the annotation (slightly above the bounding box)
        text_x = int(x + w / 2)
        text_y = max(0, y - offset)  # Ensure text doesn't go out of bounds

        # Add text annotation for the footing's dimension
        cv2.putText(annotation_layer, f"{dimension_cm} cm", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 1, cv2.LINE_AA)

    # Save the annotation layer as a PNG with transparency
    cv2.imwrite(output_path, annotation_layer)
    print(f"Footing annotation layer saved at {output_path}")
    
def create_expanded_column_annotation_layer(expanded_column_image, output_path, conversion_factor=1, offset=10):
    """
    Create an annotation layer for the expanded columns by calculating their dimensions.
    """
    # Load the expanded column image (footing layer) with transparency
    expanded_column_img = cv2.imread(expanded_column_image, cv2.IMREAD_UNCHANGED)

    if expanded_column_img is None:
        raise ValueError(f"Could not load image at {expanded_column_image}")

    # Create a binary mask from the alpha channel where expanded columns are detected
    alpha_channel = expanded_column_img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)

    # Find contours of the expanded columns
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty transparent image for the annotation layer
    annotation_layer = np.zeros((expanded_column_img.shape[0], expanded_column_img.shape[1], 4), dtype=np.uint8)

    # Iterate through each expanded column and calculate its dimensions
    for contour in contours:
        # Calculate the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate dimension in cm using the conversion factor (assuming columns are squares)
        dimension_cm = round(w * conversion_factor, 2)

        # Calculate the position for the annotation (slightly above the bounding box)
        text_x = int(x + w / 2)
        text_y = max(0, y - offset)  # Ensure text doesn't go out of bounds

        # Add text annotation for the expanded column's dimension
        cv2.putText(annotation_layer, f"{dimension_cm} cm", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 1, cv2.LINE_AA)

    # Save the annotation layer as a PNG with transparency
    cv2.imwrite(output_path, annotation_layer)
    print(f"Expanded column annotation layer saved at {output_path}")

def draw_dashed_line(img, start_point, end_point, color=(0, 0, 0, 255), thickness=1, dash_length=20, gap_length=5):
    """
    Draw a dashed line on an image from start_point to end_point in black.
    """
    # Calculate the total distance between the start and end points
    distance = int(np.linalg.norm(np.array(end_point) - np.array(start_point)))

    # Calculate the number of dashes and gaps that fit into the line segment
    num_dashes = distance // (dash_length + gap_length)

    # Iterate through the number of dashes and draw each one
    for i in range(num_dashes + 1):
        # Calculate the start and end points of each dash
        start_ratio = i * (dash_length + gap_length) / distance
        end_ratio = min(1, (i * (dash_length + gap_length) + dash_length) / distance)

        dash_start = (
            int(start_point[0] + (end_point[0] - start_point[0]) * start_ratio),
            int(start_point[1] + (end_point[1] - start_point[1]) * start_ratio)
        )
        dash_end = (
            int(start_point[0] + (end_point[0] - start_point[0]) * end_ratio),
            int(start_point[1] + (end_point[1] - start_point[1]) * end_ratio)
        )

        # Draw the dash
        cv2.line(img, dash_start, dash_end, color, thickness)

def create_dashed_grid_cross_layer(json_file, output_path, canvas_width, canvas_height, conversion_factor=6.923):
    """
    Create a grid-like cross marker layer with black dashed lines based on the column center points from the JSON file.
    """
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Create an empty transparent image for the grid cross marker layer
    grid_cross_layer = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Draw dashed grid-like crosses for each column's center point
    for column in data['columns']:
        # Calculate the center of the column
        start_x, start_y = column['start_point']
        length_px = int(column['length_cm'] / conversion_factor)  # Convert length to pixels
        height_px = int(column['height_cm'] / conversion_factor)  # Convert height to pixels

        # Correctly calculate the center point
        center_x = start_x + length_px // 2
        center_y = start_y + height_px // 2

        # Draw a dashed vertical line spanning the entire height
        draw_dashed_line(grid_cross_layer, (center_x, 0), (center_x, canvas_height), (0, 0, 0, 255))

        # Draw a dashed horizontal line spanning the entire width
        draw_dashed_line(grid_cross_layer, (0, center_y), (canvas_width, center_y), (0, 0, 0, 255))

    # Save the grid cross marker layer with dashed lines as a PNG with transparency
    cv2.imwrite(output_path, grid_cross_layer)
    print(f"Dashed grid cross marker layer saved at {output_path}")


def draw_dashed_outline(img, contours, color=(0, 0, 0, 255), thickness=1, dash_length=20, gap_length=5):
    """
    Draw a dashed outline around the specified contours.
    """
    for contour in contours:
        # Iterate over the contour points and draw dashes
        for i in range(len(contour) - 1):
            pt1 = tuple(contour[i][0])
            pt2 = tuple(contour[i + 1][0])

            # Calculate the distance between the points
            distance = int(np.linalg.norm(np.array(pt2) - np.array(pt1)))

            # Calculate the number of dashes and gaps along the segment
            num_dashes = distance // (dash_length + gap_length)

            for j in range(num_dashes + 1):
                start_ratio = j * (dash_length + gap_length) / distance
                end_ratio = min(1, (j * (dash_length + gap_length) + dash_length) / distance)

                dash_start = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                    int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)
                )
                dash_end = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                    int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)
                )

                # Draw the dash
                cv2.line(img, dash_start, dash_end, color, thickness)
                         
            
def combine_layers_and_add_dashed_outline(footing_path, padded_walls_path, output_path):
    """
    Combine the padded walls and footing layers, then add a dashed outline for both outer and inner contours.
    """
    # Load the footing and padded wall images
    footing_img = cv2.imread(footing_path, cv2.IMREAD_UNCHANGED)
    padded_walls_img = cv2.imread(padded_walls_path, cv2.IMREAD_UNCHANGED)

    if footing_img is None or padded_walls_img is None:
        raise ValueError("Could not load one or both of the images.")

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
    cv2.imwrite(output_path, combined_layer)
    print(f"Combined layer with dashed outline saved at {output_path}")


def overlay_images(base_image, overlay_image):
    """
    Overlay one image on top of another, handling transparency.
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

def combine_all_layers(layers, output_path, canvas_size):
    """
    Combine all specified layers from bottom to top with a white background.
    """
    width, height = canvas_size

    # Create a white background image with transparency
    final_image = np.zeros((height, width, 4), dtype=np.uint8)
    final_image[:, :, 0:3] = 255  # Set RGB to white
    final_image[:, :, 3] = 255  # Set alpha to fully opaque

    # Overlay each layer in reverse order (bottom to top)
    for layer_path in reversed(layers):  # Reversed order
        layer_img = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)

        if layer_img is None:
            raise ValueError(f"Could not load image at {layer_path}")

        # Ensure the layer image matches the canvas size
        if layer_img.shape[:2] != (height, width):
            layer_img = cv2.resize(layer_img, (width, height))

        # Overlay the layer on the final image
        overlay_images(final_image, layer_img)

    # Save the final combined image as a PNG
    cv2.imwrite(output_path, final_image)
    print(f"Final combined image saved at {output_path}")    
    


if __name__ == "__main__":
    column_image = 'generated_columns.png'  # Input column image
    expanded_output_path = 'expanded_columns.png'  # Output styled column layer

    expansion_amount = 10  # Expansion amount in pixels
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the expanded column

    # Expand the columns
    expand_columns(column_image, expanded_output_path, expansion_amount)
    
    footing_output = 'footing_layer_no_outline.png'  # Output footing layer
    expansion_amount = 50  # Example expansion amount in pixels

    # Create the footing layer by expanding the columns
    create_footing_layer(column_image, footing_output, expansion_amount)
    
    wall_image_path = 'generated_walls.png'  # Input wall image
    output_path = 'padded_walls_no_outline.png'  # Output padded wall layer
    padding_amount = 10  # Example padding amount in pixels

    # Add padding to the wall lines
    add_padding_to_walls(wall_image_path, output_path, padding_amount)
    
    
    json_file = 'RLFoundation.json'  # Input JSON file
    output_path = 'wall_annotation_layer.png'  # Output annotation layer
    conversion_factor = 1  # Conversion factor for units (e.g., 6.923 cm/px)

    # Create the wall annotation layer from the JSON data
    create_wall_annotation_layer(json_file, output_path, conversion_factor)
    
    
    expanded_column_image = 'footing_layer_no_outline.png'  # Input expanded column (footing) image
    output_path = 'manual_footing_annotation_layer.png'  # Output annotation layer
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the footing

    # Create the footing annotation layer by manually calculating the dimensions
    create_manual_footing_annotation_layer(expanded_column_image, output_path, conversion_factor, offset)
    
    
    column_image = 'generated_columns.png'  # Input column image
    annotation_output_path = 'manual_column_annotation_layer.png'  # Output annotation layer
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the column

    # Create the annotation layer for the expanded columns
    create_expanded_column_annotation_layer(expanded_output_path, annotation_output_path, conversion_factor, offset)
    
    
    
    json_file = '../RL/RLFoundation.json'  # Input JSON file
    output_path = 'black_dashed_grid_cross_marker_layer.png'  # Output grid cross marker layer
    canvas_width = 1346  # Example canvas width from JSON features
    canvas_height = 1106  # Example canvas height from JSON features

    # Create the grid-like cross marker layer with black dashed lines from the JSON data
    create_dashed_grid_cross_layer(json_file, output_path, canvas_width, canvas_height)
    
    footing_path = 'footing_layer_no_outline.png'  # Input footing layer
    padded_walls_path = 'padded_walls_no_outline.png'  # Input padded walls layer
    output_path = 'combined_layer_with_dashed_outline.png'  # Output combined layer

    # Combine layers and add a dashed outline
    combine_layers_and_add_dashed_outline(footing_path, padded_walls_path, output_path)
    
    layers = [
        'output/vae/wall_annotation_layer.png',  # Topmost layer
        'output/vae/manual_footing_annotation_layer.png',
        'output/vae/manual_column_annotation_layer.png',
        'output/vae/black_dashed_grid_cross_marker_layer.png',
        'output/vae/expanded_columns.png',
        'output/vae/combined_layer_with_dashed_outline.png'  # Bottommost layer
    ]

    output_path = 'final_combined_image.png'  # Output final image
    canvas_size = (1346, 1106)  # Example canvas size from JSON features

    # Combine all layers from top to bottom with a white background
    combine_all_layers(layers, output_path, canvas_size)