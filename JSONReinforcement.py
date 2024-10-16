import json
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from RL_Module import preprocess_floor_data, generate_foundation_plan, visualize_foundation_plan

# Set directories and paths
floor_folder = 'output/yolov8'
model_path = 'models/reinforcement_learning/rl_model'
output_json_path =  'output/RL/RLFoundation.json'

def load_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_data = []

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file)) as f:
            data = json.load(f)
            json_data.append((json_file, data))

    return json_data

def save_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved foundation plan to {file_path}")

def mainReinforcement():
    # Load the trained model
    model = PPO.load(model_path)
    print("Model loaded successfully.")

    # Preprocess the floor plan data
    floor_plan_features, reference_walls = preprocess_floor_data(floor_folder)
    
    # Generate the foundation plan using the model and the first floor plan as an example
    generated_plan = generate_foundation_plan(model, floor_plan_features[0], reference_walls[0])

    # Save the generated plan to a JSON file
    save_json_file(generated_plan, output_json_path)

if __name__ == "_main_":
    mainReinforcement()