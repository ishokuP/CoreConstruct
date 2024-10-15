import os
import json
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

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

    # Cast coordinates to integers before resizing
    img = img.resize((int(x2 - x1), int(y2 - y1)))  # Resize image to fit between start and end points

    # Paste the image on the layout at the specified location
    layout.paste(img, (x1, y1, int(x2), int(y2)))


# Function to calculate the canvas size based on the min/max coordinates in the JSON
def calculate_canvas_size(data):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Find the min and max coordinates from walls and columns
    for wall in data['walls']:
        start_x, start_y = wall['start_point']
        end_x, end_y = wall['end_point']
        min_x, min_y = min(min_x, start_x, end_x), min(min_y, start_y, end_y)
        max_x, max_y = max(max_x, start_x, end_x), max(max_y, start_y, end_y)

    for column in data['columns']:
        start_x, start_y = column['start_point']
        length = column['length_cm']
        height = column['height_cm']
        end_x, end_y = start_x + length, start_y + height
        min_x, min_y = min(min_x, start_x, end_x), min(min_y, start_y, end_y)
        max_x, max_y = max(max_x, start_x, end_x), max(max_y, start_y, end_y)

    # Calculate canvas size, adding a margin if necessary
    width = int(max_x - min_x)
    height = int(max_y - min_y)
    
    return width, height, min_x, min_y


# Generate separate wall and column images based on JSON coordinates
def generate_separate_images(json_file):
    model="models/vae_final.pth"
    model.eval()  # Set model to evaluation mode
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Calculate the dynamic canvas size
    canvas_width, canvas_height, offset_x, offset_y = calculate_canvas_size(data)

    # Create blank canvases for walls and columns
    wall_layout = Image.new('L', (canvas_width, canvas_height), 255)
    column_layout = Image.new('L', (canvas_width, canvas_height), 255)

    # Generate and place walls
    for wall in data['walls']:
        start_point = (wall['start_point'][0] - offset_x, wall['start_point'][1] - offset_y)
        end_point = (wall['end_point'][0] - offset_x, wall['end_point'][1] - offset_y)

        z = torch.randn(1, 128).cuda()  # Random latent vector
        with torch.no_grad():
            generated_wall = model.decode(z).squeeze().cpu().numpy()  # Generate wall

        paste_image(wall_layout, generated_wall, start_point, end_point)  # Resize and paste wall

    # Generate and place columns
    for column in data['columns']:
        start_point = (column['start_point'][0] - offset_x, column['start_point'][1] - offset_y)
        end_point = (start_point[0] + column['length_cm'], start_point[1] + column['height_cm'])

        z = torch.randn(1, 128).cuda()  # Random latent vector
        with torch.no_grad():
            generated_column = model.decode(z).squeeze().cpu().numpy()  # Generate column

        paste_image(column_layout, generated_column, start_point, end_point)  # Resize and paste column

    # Save the separate wall and column images
    wall_output_path = 'wall_layout.png'
    column_output_path = 'column_layout.png'
    wall_layout.save(wall_output_path)
    column_layout.save(column_output_path)
    print(f'Saved wall image at {wall_output_path}')
    print(f'Saved column image at {column_output_path}')


# Main script for generating separate images
if __name__ == "__main__":
    model_path = 'models/vae_final.pth'  # Path to the trained VAE model
    json_file = 'testfiles/left (3).json'  # Example JSON file

    # Initialize VAE and load trained weights
    model = ConditionalVAE(img_channels=1, latent_dim=128).cuda()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()

    # Generate separate wall and column images from the trained VAE model and JSON data
    generate_separate_images(json_file)
