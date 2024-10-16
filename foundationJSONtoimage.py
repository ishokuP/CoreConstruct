import os
import json
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from post import expandImg, style_columns, combine_images, add_dashed_outline, overlay_images_with_background, overlay_pre_generated_image

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
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
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
    img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to PIL image

    # Calculate the bounding box for the image placement
    x1, y1 = start_point
    if end_point:
        x2, y2 = end_point
    else:
        x2, y2 = x1 + img.width, y1 + img.height

    # Ensure that the width and height are positive
    width = max(1, int(x2 - x1))
    height = max(1, int(y2 - y1))

    # Resize the image to fit the bounding box
    img = img.resize((width, height))

    # Create an alpha mask for transparency
    img = img.convert("RGBA")
    alpha_mask = img.split()[-1]

    # Paste using the alpha mask to maintain transparency
    layout.paste(img, (x1, y1), mask=alpha_mask)


# Function to calculate the canvas size based on the min/max coordinates in the JSON and add padding
def calculate_canvas_size(data):
    """
    Retrieves the canvas size from the 'features' section of the JSON data.
    """
    # Check if 'features' and 'canvas_width'/'canvas_height' exist in the JSON
    if 'features' in data and 'canvas_width' in data['features'] and 'canvas_height' in data['features']:
        canvas_width = data['features']['canvas_width']
        canvas_height = data['features']['canvas_height']
        return (canvas_width, canvas_height, 0, 0)
    else:
        raise ValueError("Canvas dimensions are not defined in the 'features' section of the JSON data.")


def generate_annotation_layer(data, canvas_width, canvas_height):
    """
    Creates an annotation layer with wall lengths displayed at midpoints.
    """
    annotation_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(annotation_layer)

    # Specify the font size
    font_size = 20  # Adjust this value as needed for larger or smaller text

    # Load a default PIL font with the specified size
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Requires arial.ttf installed
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if arial is not available

    # Iterate through the walls in the JSON data
    for wall in data['walls']:
        start_point = wall['start_point']
        end_point = wall['end_point']
        length_cm = wall['length_cm']

        # Calculate the midpoint of the wall
        midpoint = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)

        # Annotate the wall length on the image at the midpoint
        draw.text(midpoint, f"{length_cm:.2f} cm", fill=(0, 0, 0, 255), font=font)

    # Save the annotation layer as a PNG
    annotation_output_path = 'output/vae/annotation_layer.png'
    annotation_layer.save(annotation_output_path, "PNG")
    print(f'Annotation layer saved at {annotation_output_path}')
    return annotation_output_path


# Generate separate wall and column images based on JSON coordinates
def generate_separate_images(json_file, column_expansion_amount, wall_expansion_amount,column_scale):
    # Load and initialize the model inside the function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConditionalVAE(img_channels=1, latent_dim=128).to(device)
    model_path = 'models/vae/vae_final.pth'  # Define the path to the model here
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Calculate the dynamic canvas size with padding
    canvas_width, canvas_height, offset_x, offset_y = calculate_canvas_size(data)

    # Create blank canvases for walls and columns with transparency
    wall_layout = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))  # Fully transparent background
    column_layout = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))  # Fully transparent background

    # Generate and place walls
    for wall in data['walls']:
        start_point = (wall['start_point'][0] - offset_x, wall['start_point'][1] - offset_y)
        end_point = (wall['end_point'][0] - offset_x, wall['end_point'][1] - offset_y)

        z = torch.randn(1, 128).to(device)
        with torch.no_grad():
            generated_wall = model.decode(z).squeeze().cpu().numpy()

        paste_image(wall_layout, generated_wall, start_point, end_point)

    # Generate and place columns
    for column in data['columns']:
        start_point = (column['start_point'][0] - offset_x, column['start_point'][1] - offset_y)
        end_point = (start_point[0] + column['length_cm'], start_point[1] + column['height_cm'])

        z = torch.randn(1, 128).to(device)
        with torch.no_grad():
            generated_column = model.decode(z).squeeze().cpu().numpy()

        paste_image(column_layout, generated_column, start_point, end_point)
        
        
    # Generate annotation layer
    annotation_output_path = generate_annotation_layer(data, canvas_width, canvas_height)


    # Save the separate wall and column images
    wall_output_path = 'output/vae/wall_layout.png'
    column_output_path = 'output/vae/column_layout.png'
    wall_layout.save(wall_output_path, "PNG")
    column_layout.save(column_output_path, "PNG")
    print(f'Saved wall image at {wall_output_path}')
    print(f'Saved column image at {column_output_path}')
    print(f'Saved annotation image at {annotation_output_path}')
    
    
    
    wall_layout_path = 'output/vae/wall_layout.png'  # Path to the wall layout image
    column_layout_path = 'output/vae/column_layout.png'  # Path to the column layout image
    footing_output_path = 'output/vae/footing.png'  # Path to save the footing image
    expanded_wall_output_path = 'output/vae/expanded_wall.png'  # Path to save the expanded wall image
    styled_column_output_path = 'output/vae/styled_column_layout.png'  # Path to save the styled column image
    combined_output_path = 'output/vae/combined_layout.png'  # Path to save the combined image
    dashed_output_path = 'output/vae/combined_layout_dashed.png'  # Path to save the image with dashed outline
    final_output_path = 'static/images/final_combined.png'  # Path to save the final image with the white background
    annotation_layer_path = 'output/vae/annotation_layer.png'  # Pre-generated annotation layer
    annotated_output_path = 'static/images/final_combined.png'  # Path for annotated image


    # Expansion amounts (can be adjusted as needed)

    # Create the expanded wall image (similar to footing)
    expandImg(wall_layout_path, expanded_wall_output_path, wall_expansion_amount)

    # Create the footing from the column layout
    expandImg(column_layout_path, footing_output_path, column_expansion_amount)

    # Style the columns with white fill and black outline
    style_columns(column_layout_path, styled_column_output_path, column_scale)

    # Combine the footing and expanded wall images
    combine_images(footing_output_path, expanded_wall_output_path, combined_output_path)

    # Add a dashed outline to the combined image
    add_dashed_outline(combined_output_path, dashed_output_path)

    # Overlay the styled columns on top of the dashed image and add a white background
    overlay_images_with_background(dashed_output_path, styled_column_output_path, final_output_path)
    
    # Overlay the pre-generated annotation layer on top of the final combined image
    overlay_pre_generated_image(final_output_path, annotation_layer_path, annotated_output_path)
