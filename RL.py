import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

# Import the environment and helper functions
from rl_main import FoundationDesignEnv, preprocess_all_data, generate_foundation_plan, visualize_foundation_plan

def load_trained_model(model_path):
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_floor_plan_files(folder_path):

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    floor_plan_data = []

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file)) as f:
            data = json.load(f)
            floor_plan_data.append((json_file, data))

    return floor_plan_data

def preprocess_floor_data(floor_folder):

    floor_data = load_floor_plan_files(floor_folder)

    floor_plan_features = []
    reference_walls = []

    for (floor_filename, floor_plan) in floor_data:
        # Extract midpoints, corners, and walls from the floor plan
        midpoints = floor_plan.get('averaged_midpoints', [])
        corners = floor_plan.get('averaged_corners', [])
        walls = floor_plan.get('walls', [])

        # Create features from midpoints and corners
        features = [{'x': point['x'], 'y': point['y'], 'type': 'midpoint'} for point in midpoints] + \
                   [{'x': point['x'], 'y': point['y'], 'type': 'corner'} for point in corners]

        # Add reference walls
        reference_walls.append(walls)

        # Append the features to the list
        floor_plan_features.append(features)

    print(f"Total floor plans processed: {len(floor_plan_features)}")
    return floor_plan_features, reference_walls

def test_model_on_floor_plan(model, floor_plan_features, reference_walls, index=0, save_path="generated_plan.json"):
    """
    Test the model on a given floor plan and save the generated plan to a JSON file.
    """
    # Select the floor plan features and reference walls at the specified index
    floor_features = floor_plan_features[index]
    ref_walls = reference_walls[index]

    # Create a new environment with the floor features and empty foundation plan
    env = FoundationDesignEnv(floor_features, {'columns': [], 'walls': []}, ref_walls)
    obs, info = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Extract the generated foundation plan
    generated_plan = {
        'columns': env.predicted_plan['columns'],
        'walls': env.predicted_plan['walls']
    }

    # Save the generated plan to a JSON file
    try:
        with open(save_path, 'w') as f:
            json.dump(generated_plan, f, indent=4)
        print(f"Generated plan saved to {save_path}")
    except Exception as e:
        print(f"Error saving generated plan: {e}")

    return generated_plan

def rl_module():
    # Set the paths to the floor folder and model
    floor_folder = 'output/yolov8/cleanedJSON'  # Adjust this path based on your dataset location
    model_path = "models/reinforcement_learning/rl_model"  # Path to your saved model

    # Load the trained model
    model = load_trained_model(model_path)
    if model is None:
        return

    # Preprocess the floor plan data, including reference walls
    floor_plan_features, reference_walls = preprocess_floor_data(floor_folder)
    if not floor_plan_features:
        print("No floor plans found. Exiting.")
        return

    # Test the model on the first available floor plan and save the output
    index = 0  # Adjust the index as needed
    generated_plan = test_model_on_floor_plan(
        model, floor_plan_features, reference_walls, index=index, save_path="output/RL/RLFoundation.json"
    )

    # # Optional: visualize the generated plan
    # visualize_foundation_plan(generated_plan)

if __name__ == "__main__":
    rl_module()
