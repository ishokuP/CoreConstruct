import cv2
import numpy as np
from PIL import Image

def create_expanded_image(input_image_path, output_image_path, expansion_amount, fill_color=(255, 255, 255, 255)):
    """
    General function to expand regions in an image and fill them with a specified color.
    This function is used for creating both the wall and footing images.
    """
    # Read the input image using OpenCV
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    # Create a binary mask where black pixels are foreground and the rest is background
    mask = cv2.inRange(input_image, (0, 0, 0, 255), (0, 0, 0, 255))

    # Expand (dilate) the mask using the specified expansion amount
    kernel = np.ones((expansion_amount, expansion_amount), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel)

    # Create a filled image using the expanded mask as the alpha channel
    expanded_image = np.zeros_like(input_image)
    expanded_image[:, :, 0:3] = fill_color[0:3]  # Set RGB to the fill color
    expanded_image[:, :, 3] = expanded_mask  # Alpha channel from the expanded mask

    # Save the expanded image as a PNG file
    cv2.imwrite(output_image_path, expanded_image)
    print(f'Expanded image saved at {output_image_path}')

def style_columns(column_image_path, output_column_path):
    """
    Function to style columns with a white fill and black outline.
    """
    # Read the column layout image using OpenCV
    column_image = cv2.imread(column_image_path, cv2.IMREAD_UNCHANGED)

    # Create a binary mask where black pixels (columns) are white (foreground)
    column_mask = cv2.inRange(column_image, (0, 0, 0, 255), (0, 0, 0, 255))

    # Find contours of the columns
    contours, _ = cv2.findContours(column_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image for the styled columns with transparency
    styled_columns = np.zeros_like(column_image)

    # Draw the columns with a white fill and black outline
    for contour in contours:
        cv2.drawContours(styled_columns, [contour], -1, (0, 0, 0, 255), 2)  # Black outline
        cv2.drawContours(styled_columns, [contour], -1, (255, 255, 255, 255), -1)  # White fill

    # Save the styled columns image as PNG
    cv2.imwrite(output_column_path, styled_columns)
    print(f'Styled columns image saved at {output_column_path}')

def combine_images(footing_image_path, wall_image_path, output_path):
    """
    Function to combine the footing and expanded wall images.
    """
    # Load the images
    footing_image = cv2.imread(footing_image_path, cv2.IMREAD_UNCHANGED)
    wall_image = cv2.imread(wall_image_path, cv2.IMREAD_UNCHANGED)

    # Check if the images are loaded properly
    if footing_image is None:
        raise ValueError(f"Could not load image at {footing_image_path}")
    if wall_image is None:
        raise ValueError(f"Could not load image at {wall_image_path}")

    # Ensure the images are the same size
    if footing_image.shape != wall_image.shape:
        # Resize the wall image to match the size of the footing image
        wall_image = cv2.resize(wall_image, (footing_image.shape[1], footing_image.shape[0]))

    # Combine the images by overlaying (using transparency if present)
    combined_image = np.zeros_like(footing_image)

    # If images have alpha channels (transparency), blend them
    if footing_image.shape[2] == 4 and wall_image.shape[2] == 4:
        alpha_footing = footing_image[:, :, 3] / 255.0
        alpha_wall = wall_image[:, :, 3] / 255.0

        for c in range(0, 3):  # Loop through the color channels (B, G, R)
            combined_image[:, :, c] = (footing_image[:, :, c] * alpha_footing +
                                       wall_image[:, :, c] * alpha_wall * (1 - alpha_footing))

        # Update the alpha channel
        combined_image[:, :, 3] = np.clip(alpha_footing + alpha_wall * (1 - alpha_footing), 0, 1) * 255

    else:
        # If no alpha channel is present, use simple addition or averaging
        combined_image = cv2.addWeighted(footing_image, 0.5, wall_image, 0.5, 0)

    # Save the combined image
    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved at {output_path}")

def add_dashed_outline(image_path, output_path, color=(0, 0, 0, 255), dash_length=10, gap_length=10):
    """
    Function to add a dashed line outline to the combined image.
    """
    # Load the combined image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to grayscale and create a binary mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and draw dashed lines
    for contour in contours:
        # Iterate over each point in the contour to draw dashed lines
        for i in range(len(contour) - 1):
            pt1 = tuple(contour[i][0])
            pt2 = tuple(contour[i + 1][0])

            # Calculate the total distance between the points
            distance = np.linalg.norm(np.array(pt2) - np.array(pt1))

            # Calculate the number of dashes and gaps that fit into the line segment
            num_dashes = int(distance // (dash_length + gap_length))

            # Draw the dashed line
            for j in range(num_dashes + 1):
                start_ratio = j * (dash_length + gap_length) / distance
                end_ratio = (j * (dash_length + gap_length) + dash_length) / distance

                if end_ratio > 1:
                    end_ratio = 1

                dash_start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
                dash_end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio), int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))

                # Draw the dash
                cv2.line(image, dash_start, dash_end, color, 1, cv2.LINE_AA)

    # Save the image with the dashed outline
    cv2.imwrite(output_path, image)
    print(f'Dashed outline image saved at {output_path}')

def overlay_images_with_background(base_image_path, overlay_image_path, output_path, background_color=(255, 255, 255)):
    """
    Function to overlay the styled column image on top of the dashed image with a white background.
    """
    # Load the base (dashed outline) image and the overlay (styled column) image
    base_image = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

    # Check if the images are loaded properly
    if base_image is None:
        raise ValueError(f"Could not load image at {base_image_path}")
    if overlay_image is None:
        raise ValueError(f"Could not load image at {overlay_image_path}")

    # Ensure the images are the same size
    if base_image.shape != overlay_image.shape:
        overlay_image = cv2.resize(overlay_image, (base_image.shape[1], base_image.shape[0]))

    # Create a white background image
    background = np.ones_like(base_image) * 255  # Create a white background (RGB + Alpha)

    # Overlay the base (dashed outline) image on the white background
    if base_image.shape[2] == 4:
        alpha_base = base_image[:, :, 3] / 255.0
        for c in range(0, 3):  # Iterate over the B, G, R channels
            background[:, :, c] = (background[:, :, c] * (1 - alpha_base) +
                                   base_image[:, :, c] * alpha_base)

    # Overlay the styled column image on top of the combined image (base + background)
    if overlay_image.shape[2] == 4:
        alpha_overlay = overlay_image[:, :, 3] / 255.0
        for c in range(0, 3):  # Iterate over the B, G, R channels
            background[:, :, c] = (background[:, :, c] * (1 - alpha_overlay) +
                                   overlay_image[:, :, c] * alpha_overlay)

    # Save the final combined image
    cv2.imwrite(output_path, background)
    print(f'Final combined image with styled columns saved at {output_path}')


if __name__ == "__main__":
    # Paths to input images and output files
    wall_layout_path = 'output/vae/wall_layout.png'  # Path to the wall layout image
    column_layout_path = 'output/vae/column_layout.png'  # Path to the column layout image
    footing_output_path = 'output/vae/footing.png'  # Path to save the footing image
    expanded_wall_output_path = 'output/vae/expanded_wall.png'  # Path to save the expanded wall image
    styled_column_output_path = 'output/vae/styled_column_layout.png'  # Path to save the styled column image
    combined_output_path = 'output/vae/combined_layout.png'  # Path to save the combined image
    dashed_output_path = 'output/vae/combined_layout_dashed.png'  # Path to save the image with dashed outline
    final_output_path = 'output/vae/final_combined.png'  # Path to save the final image with the white background

    # Expansion amounts (can be adjusted as needed)
    column_expansion_amount = 50
    wall_expansion_amount = 10

    # Create the expanded wall image (similar to footing)
    create_expanded_image(wall_layout_path, expanded_wall_output_path, wall_expansion_amount)

    # Create the footing from the column layout
    create_expanded_image(column_layout_path, footing_output_path, column_expansion_amount)

    # Style the columns with white fill and black outline
    style_columns(column_layout_path, styled_column_output_path)

    # Combine the footing and expanded wall images
    combine_images(footing_output_path, expanded_wall_output_path, combined_output_path)

    # Add a dashed outline to the combined image
    add_dashed_outline(combined_output_path, dashed_output_path)

    # Overlay the styled columns on top of the dashed image and add a white background
    overlay_images_with_background(dashed_output_path, styled_column_output_path, final_output_path)