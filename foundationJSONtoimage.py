import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from post import add_padding_to_json, create_footing_info_layer, expand_columns, create_dashed_grid_cross_layer, create_footing_layer, add_padding_to_walls, create_manual_footing_annotation_layer, create_wall_annotation_layer, create_expanded_column_annotation_layer, combine_all_layers, combine_layers_and_add_dashed_outline
# TODO: post
# Conditional VAE Model (same as before)


class ConditionalVAE(nn.Module):
    def __init__(self, img_channels=1, latent_dim=128):
        super(ConditionalVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z).view(-1, 512, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Helper function to paste images on the layout with appropriate resizing


def paste_image(layout, img, start_point, end_point=None):
    # Ensure start and end points are integers
    x1, y1 = map(int, start_point)
    x2, y2 = map(int, end_point)

    # Calculate width and height for resizing
    width = max(1, x2 - x1)  # Ensure width is at least 1
    height = max(1, y2 - y1)  # Ensure height is at least 1

    # Debugging: Print the dimensions
    print(f"Resizing image to: Width = {width}, Height = {height}")

    # Resize image to fit within the specified region
    img = (img * 255).astype(np.uint8)  # Convert to 8-bit

    # Check if img is a valid image
    if img is None or img.size == 0:
        print("Error: Image is empty or None.")
        return

    # Resize image with OpenCV
    try:
        img = cv2.resize(img, (width, height))
    except cv2.error as e:
        print(f"Error resizing image: {e}")
        return

    # Ensure the image has an alpha channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    # Ensure the layout has sufficient dimensions
    if layout.shape[0] >= y2 and layout.shape[1] >= x2:
        # Paste the image on the layout at the specified location
        layout[y1:y2, x1:x2, :] = img
    else:
        print("Error: Layout dimensions are smaller than the paste region.")

# Generate solid rectangles for walls


def draw_solid_wall(layout, start_point, end_point):
    """
    Draw a solid wall as a filled rectangle.
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # Draw a filled rectangle (white) with a black border
    cv2.rectangle(layout, (x1, y1), (x2, y2),
                  (255, 255, 255, 255), thickness=-1)  # Fill
    cv2.rectangle(layout, (x1, y1), (x2, y2),
                  (0, 0, 0, 255), thickness=2)  # Border

# Generate separate images for walls and columns based on the JSON file

def add_padding(image, padding_percentage=0.3):
    """
    Adds padding around the image by a given percentage.
    """
    height, width = image.shape[:2]

    # Calculate padding dimensions
    pad_h = int(height * padding_percentage)
    pad_w = int(width * padding_percentage)

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )
    return padded_image, pad_w, pad_h


def generate_layout(model, json_file):
    model.eval()  # Set model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Get fixed canvas width and height from JSON features
    canvas_width = data['features']['canvas_width']
    canvas_height = data['features']['canvas_height']

    # Create separate canvases for walls and columns (transparent background)
    wall_layout = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)
    column_layout = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    # Generate and place walls
    for wall in data['walls']:
        start_point = (wall['start_point'][0], wall['start_point'][1])
        end_point = (wall['end_point'][0], wall['end_point'][1])

        # Draw a solid wall rectangle
        draw_solid_wall(wall_layout, start_point, end_point)

    # Generate and place columns
    for column in data['columns']:
        start_point = (column['start_point'][0], column['start_point'][1])
        end_point = (column['end_point'][0], column['end_point'][1])

        z = torch.randn(1, 128, device=device)  # Adjusted for device compatibility
        with torch.no_grad():
            generated_column = model.decode(
                z).squeeze().cpu().numpy()  # Generate column

        paste_image(column_layout, generated_column, start_point,
                    end_point)  # Paste column on the layout

    # Add 30% padding to the wall and column layouts
    wall_layout_padded, pad_w, pad_h = add_padding(wall_layout, padding_percentage=0.2)
    column_layout_padded, _, _ = add_padding(column_layout, padding_percentage=0.2)
    
    
    # Save wall and column layouts as separate images with transparency
    wall_output_path = 'output/vae/generated_walls.png'
    column_output_path = 'output/vae/generated_columns.png'
    cv2.imwrite(wall_output_path, wall_layout_padded)
    cv2.imwrite(column_output_path, column_layout_padded)

    print(f'Wall layout saved at {wall_output_path}')
    print(f'Column layout saved at {column_output_path}')

# Load the trained VAE model and generate layout based on JSON


def generateFoundationPlan(json_file,column_scale, footing_scale,num_storey_value,barsize_value,model_path='models/vae/vae_final.pth'):
    # Initialize VAE and load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConditionalVAE(img_channels=1, latent_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate layout based on the input JSON file
    generate_layout(model, json_file)


    # Input and output file paths
    column_image = 'output/vae/generated_columns.png'
    expanded_columns_output = 'output/vae/expanded_columns.png'
    footing_output = 'output/vae/footing_layer_no_outline.png'
    wall_image_path = 'output/vae/generated_walls.png'
    padded_walls_output = 'output/vae/padded_walls_no_outline.png'
    json_file_path = 'output/RL/RLFoundation.json'
    padded_json_output = 'output/RL/RLFoundationPadded.json'
    wall_annotation_output = 'output/vae/wall_annotation_layer.png'
    footing_annotation_output = 'output/vae/manual_footing_annotation_layer.png'
    column_annotation_output = 'output/vae/manual_column_annotation_layer.png'
    grid_cross_output = 'output/vae/black_dashed_grid_cross_marker_layer.png'
    combined_layer_output = 'output/vae/combined_layer_with_dashed_outline.png'
    footing_info_output = 'output/vae/footing_info_layer.png'  # New output file
    final_combined_output = 'static/images/final_combined_image.png'

    # Parameters
    expansion_amount_columns = int(10 * column_scale)
    expansion_amount_footing = int(90  * footing_scale)
    padding_amount_walls = 40
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset for annotations
    padding_percentage = 0.2  # Padding percentage for JSON data

    # Step 1: Expand the columns
    expand_columns(column_image, expanded_columns_output, expansion_amount_columns)

    # Step 2: Create the footing layer by expanding the columns
    create_footing_layer(column_image, footing_output, expansion_amount_footing)

    # Step 3: Load and modify JSON data with padding
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    padded_json_data = add_padding_to_json(json_data, padding_percentage)
    with open(padded_json_output, 'w') as f:
        json.dump(padded_json_data, f, indent=4)
    print(f"Updated JSON saved at {padded_json_output}")

    # Step 4: Add padding to the walls
    add_padding_to_walls(wall_image_path, padded_walls_output, padding_amount_walls)

    # Step 5: Create wall annotation layer
    create_wall_annotation_layer(padded_json_output, wall_annotation_output, conversion_factor=1)

    # Step 6: Create footing annotation layer and get footing size
    footing_annotation_layer, footing_size_cm = create_manual_footing_annotation_layer(footing_output, conversion_factor, offset)

    # Step 7: Create expanded column annotation layer
    create_expanded_column_annotation_layer(expanded_columns_output, column_annotation_output, conversion_factor, offset)

    # Step 8: Create dashed grid cross marker layer
    canvas_width = padded_json_data['features']['canvas_width']
    canvas_height = padded_json_data['features']['canvas_height']
    create_dashed_grid_cross_layer(padded_json_output, grid_cross_output, canvas_width, canvas_height)

    # Step 9: Combine layers and add dashed outline
    combine_layers_and_add_dashed_outline(footing_output, padded_walls_output, combined_layer_output)

    # Step 10: Create footing information layer
    reinforcement_diameter = barsize_value  # User input (in mm)
    number_of_storeys = num_storey_value  # User input (1 or 2)
    create_footing_info_layer(footing_output, footing_info_output, footing_size_cm, reinforcement_diameter, number_of_storeys, conversion_factor, offset)
    # Step 11: Combine all layers into the final image
    layers = [
        wall_annotation_output,
        footing_annotation_output,
        column_annotation_output,
        footing_info_output,          # New layer with footing info
        grid_cross_output,
        expanded_columns_output,
        combined_layer_output  # Bottommost layer
    ]
    canvas_size = (canvas_width, canvas_height)
    combine_all_layers(layers, final_combined_output, canvas_size)





# Main script for generating layout
# if __name__ == "__main__":
#     model_path = 'vae_weights/vae_final.pth'  # Path to the trained VAE model
#     json_file = 'output/RL/RLFoundation.json'  # Example JSON file

#     # Generate layout from the trained VAE model and JSON data
#     generateFoundationPlan(json_file)
