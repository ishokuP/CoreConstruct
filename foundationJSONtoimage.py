import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from post import expand_columns, create_dashed_grid_cross_layer, create_footing_layer, add_padding_to_walls, create_manual_footing_annotation_layer, create_wall_annotation_layer, create_expanded_column_annotation_layer, combine_all_layers, combine_layers_and_add_dashed_outline
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
    x1, y1 = start_point
    x2, y2 = end_point

    # Calculate width and height for resizing
    width = max(1, x2 - x1)  # Ensure width is at least 1
    height = max(1, y2 - y1)  # Ensure height is at least 1

    # Resize image to fit within the specified region
    img = (img * 255).astype(np.uint8)  # Convert to 8-bit
    img = cv2.resize(img, (width, height))

    # Ensure the image has an alpha channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    # Paste the image on the layout at the specified location
    layout[y1:y2, x1:x2, :] = img

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


def generate_layout(model, json_file):
    model.eval()  # Set model to evaluation mode
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

        z = torch.randn(1, 128).cuda()  # Random latent vector
        with torch.no_grad():
            generated_column = model.decode(
                z).squeeze().cpu().numpy()  # Generate column

        paste_image(column_layout, generated_column, start_point,
                    end_point)  # Paste column on the layout

    # Save wall and column layouts as separate images with transparency
    wall_output_path = 'output/vae/generated_walls.png'
    column_output_path = 'output/vae/generated_columns.png'
    cv2.imwrite(wall_output_path, wall_layout)
    cv2.imwrite(column_output_path, column_layout)

    print(f'Wall layout saved at {wall_output_path}')
    print(f'Column layout saved at {column_output_path}')

# Load the trained VAE model and generate layout based on JSON


def generateFoundationPlan(json_file, model_path='models/vae/vae_final.pth'):
    # Initialize VAE and load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConditionalVAE(img_channels=1, latent_dim=128).to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True)
                          )
    model.eval()

    # Generate layout based on the input JSON file
    generate_layout(model, json_file)

    column_image = 'output/vae/generated_columns.png'  # Input column image
    expanded_output_path = 'output/vae/expanded_columns.png'  # Output styled column layer




    # TODO: Honestly fix this part up its very scuffed for now
    expansion_amount = 10  # Expansion amount in pixels
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the expanded column

    # Expand the columns
    expand_columns(column_image, expanded_output_path, expansion_amount)

    footing_output = 'output/vae/footing_layer_no_outline.png'  # Output footing layer
    expansion_amount = 50  # Example expansion amount in pixels

    # Create the footing layer by expanding the columns
    create_footing_layer(column_image, footing_output, expansion_amount)

    wall_image_path = 'output/vae/generated_walls.png'  # Input wall image
    output_path = 'output/vae/padded_walls_no_outline.png'  # Output padded wall layer
    padding_amount = 10  # Example padding amount in pixels

    # Add padding to the wall lines
    add_padding_to_walls(wall_image_path, output_path, padding_amount)

    json_file = 'output/RL/RLFoundation.json'  # Input JSON file
    output_path = 'output/vae/wall_annotation_layer.png'  # Output annotation layer
    conversion_factor = 1  # Conversion factor for units (e.g., 6.923 cm/px)

    # Create the wall annotation layer from the JSON data
    create_wall_annotation_layer(json_file, output_path, conversion_factor)

    # Input expanded column (footing) image
    expanded_column_image = 'output/vae/footing_layer_no_outline.png'
    output_path = 'output/vae/manual_footing_annotation_layer.png'  # Output annotation layer
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the footing

    # Create the footing annotation layer by manually calculating the dimensions
    create_manual_footing_annotation_layer(
        expanded_column_image, output_path, conversion_factor, offset)

    column_image = 'output/vae/generated_columns.png'  # Input column image
    annotation_output_path = 'output/vae/manual_column_annotation_layer.png'  # Output annotation layer
    conversion_factor = 6.923  # Example conversion factor (cm/px)
    offset = 10  # Offset to place the annotation slightly above the column

    # Create the annotation layer for the expanded columns
    create_expanded_column_annotation_layer(
        expanded_output_path, annotation_output_path, conversion_factor, offset)
    
    
    output_path = 'output/vae/black_dashed_grid_cross_marker_layer.png'

    # Path to the input JSON file
    json_file = 'output/RL/RLFoundation.json'  # Replace with your actual file path

    # Open and read the JSON file
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{json_file}'.")
        exit(1)

    # Extract the canvas width and height from the JSON data
    canvas_width = data.get('features', {}).get('canvas_width', None)
    canvas_height = data.get('features', {}).get('canvas_height', None)

    # Create the grid-like cross marker layer with black dashed lines from the JSON data
    create_dashed_grid_cross_layer(
        json_file, output_path, canvas_width, canvas_height)

    footing_path = 'output/vae/footing_layer_no_outline.png'  # Input footing layer
    padded_walls_path = 'output/vae/padded_walls_no_outline.png'  # Input padded walls layer
    output_path = 'output/vae/combined_layer_with_dashed_outline.png'  # Output combined layer

    # Combine layers and add a dashed outline
    combine_layers_and_add_dashed_outline(
        footing_path, padded_walls_path, output_path)

    layers = [
        'output/vae/wall_annotation_layer.png',  # Topmost layer
        'output/vae/manual_footing_annotation_layer.png',
        'output/vae/manual_column_annotation_layer.png',
        'output/vae/black_dashed_grid_cross_marker_layer.png',
        'output/vae/expanded_columns.png',
        'output/vae/combined_layer_with_dashed_outline.png'  # Bottommost layer
    ]

    output_path = 'static/images/final_combined_image.png'  # Output final image
    canvas_size = (canvas_width, canvas_height)  # Example canvas size from JSON features

    # Combine all layers from top to bottom with a white background
    combine_all_layers(layers, output_path, canvas_size)


# Main script for generating layout
# if __name__ == "__main__":
#     model_path = 'vae_weights/vae_final.pth'  # Path to the trained VAE model
#     json_file = 'output/RL/RLFoundation.json'  # Example JSON file

#     # Generate layout from the trained VAE model and JSON data
#     generateFoundationPlan(json_file)
