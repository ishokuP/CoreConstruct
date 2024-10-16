import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt

def load_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_data = []
    
    for json_file in json_files:
        with open(os.path.join(folder_path, json_file)) as f:
            data = json.load(f)
            json_data.append((json_file, data))  

    return json_data



def preprocess_floor_data(floor_folder):
    floor_data = load_json_files(floor_folder)

    if not floor_data:
        print("No floor plan files found in the folder.")
        return [], []

    floor_plan_features = []
    reference_walls = []

    for floor_filename, floor_plan in floor_data:
        # Extract walls, midpoints, and corners
        midpoints = floor_plan.get('averaged_midpoints', [])  
        corners = floor_plan.get('averaged_corners', [])     
        walls = floor_plan.get('walls', [])
        
        # Compile features for the model (midpoints and corners)
        features = [{'x': point['x'], 'y': point['y'], 'type': 'midpoint'} for point in midpoints] + \
                   [{'x': point['x'], 'y': point['y'], 'type': 'corner'} for point in corners]

        # Append the features and reference walls for this floor plan
        floor_plan_features.append(features)
        reference_walls.append(walls)

    return floor_plan_features, reference_walls



class FoundationDesignEnv(gym.Env):
    def __init__(self, floor_plan_features, foundation_plan, reference_walls, max_features=50, threshold=100):
        super(FoundationDesignEnv, self).__init__()
        if not floor_plan_features:
            raise ValueError("floor_plan_features should not be empty.")
        self.reference_walls = reference_walls
        self.floor_plan_features = floor_plan_features
        self.reference_plan = foundation_plan  
        self.predicted_plan = {'columns': [], 'walls': []}  
        self.current_step = 0
        self.threshold = threshold
        self.max_features = max_features

        self.action_space = spaces.Discrete(1) 
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_features,), dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False

        if self.current_step < len(self.floor_plan_features):
            feature = self.floor_plan_features[self.current_step]

            if feature['type'] == 'corner':  # Place at corner
                self.place_column(feature)
                reward += 1
            elif feature['type'] == 'midpoint':  # Handle midpoint placement
                matches_reference_column = any(
                    self.is_matching_column(feature, ref_col)
                    for ref_col in self.reference_plan['columns']
                )
                if matches_reference_column:
                    self.place_column(feature)
                    reward += 0.75
                else:
                    reward -= 0.5

            column_count = len(self.predicted_plan['columns'])
            reference_wall_count = len(self.reference_walls)
            if column_count >= 2:
                for i in range(len(self.predicted_plan['columns']) - 1):
                    for j in range(i + 1, len(self.predicted_plan['columns'])):
                        col1 = self.predicted_plan['columns'][i]
                        col2 = self.predicted_plan['columns'][j]

                        # First, try to match walls based on reference
                        matching_wall = any(
                            self.is_matching_wall({'start': col1, 'end': col2}, wall)
                            for wall in self.reference_walls
                        )
                        if matching_wall:
                            self.place_wall(col1, col2)
                            reward += 2  # Reward for placing a valid wall
                        else:
                            # Fallback logic: if fewer walls placed than reference, add more walls
                            if len(self.predicted_plan['walls']) < reference_wall_count:

                                self.place_wall(col1, col2)
                                reward += 1  # Small reward for adding missing walls

            self.current_step += 1
            done = self.current_step >= len(self.floor_plan_features)
        else:
            done = True

        max_steps = len(self.floor_plan_features)
        truncated = self.current_step >= max_steps
        return self._get_obs(), reward, done, truncated, {}


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        
        self.current_step = 0
        self.predicted_plan = {'columns': [], 'walls': []}  # Reset agent's predicted plan
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.array([1 if i < self.current_step else 0 for i in range(len(self.floor_plan_features))], dtype=np.float32)
        return np.pad(obs, (0, self.max_features - len(obs)), 'constant')

    def can_connect_columns(self, col1, col2):
        ALIGNMENT_TOLERANCE = 100
        x_diff = abs(col1['start_point'][0] - col2['start_point'][0])
        y_diff = abs(col1['start_point'][1] - col2['start_point'][1])

        if x_diff <= ALIGNMENT_TOLERANCE:
            return True
        elif y_diff <= ALIGNMENT_TOLERANCE:
            return True
        else:
            return False

    def place_column(self, col):
    # Define column dimensions (for example, 25x25 cm)
      column_width = 25
      column_height = 25
      
      # Adjust the column representation to include start and end points, width, and height
      column = {
          'start_point': (col['x'], col['y']),
          'end_point': (col['x'] + column_width, col['y'] + column_height),
          'length_cm': column_width,  # Assuming width is equivalent to length for simplicity
          'height_cm': column_height,
          'type': 'column'
      }
      self.predicted_plan['columns'].append(column)

    def place_wall(self, start_col, end_col):
      pixel_length = np.sqrt((end_col['start_point'][0] - start_col['start_point'][0]) ** 2 + (end_col['start_point'][1] - start_col['start_point'][1]) ** 2) # Accessing 'x' and 'y' from 'start_point'
      length_cm = self.convert_pixels_to_cm(pixel_length)
      # Determine wall orientation and place it accordingly
      if abs(start_col['start_point'][0] - end_col['start_point'][0]) <= self.threshold:  # Vertical wall # Accessing 'x' and 'y' from 'start_point'
          wall_start = (start_col['start_point'][0], min(start_col['start_point'][1], end_col['start_point'][1])) # Accessing 'x' and 'y' from 'start_point'
          wall_end = (start_col['start_point'][0], max(start_col['start_point'][1], end_col['start_point'][1])) # Accessing 'x' and 'y' from 'start_point'
      elif abs(start_col['start_point'][1] - end_col['start_point'][1]) <= self.threshold:  # Horizontal wall # Accessing 'x' and 'y' from 'start_point'
          wall_start = (min(start_col['start_point'][0], end_col['start_point'][0]), start_col['start_point'][1]) # Accessing 'x' and 'y' from 'start_point'
          wall_end = (max(start_col['start_point'][0], end_col['start_point'][0]), start_col['start_point'][1]) # Accessing 'x' and 'y' from 'start_point'
      else:
          return  
      # Check for overlaps with existing walls
      for wall in self.predicted_plan['walls']:
          if self.is_overlapping(wall_start, wall_end, wall['start_point'], wall['end_point']):
              return  # Exit if overlap detected

      # Add the new wall if no overlap is found
      wall = {
          'start_point': wall_start,
          'end_point': wall_end,
          'length_cm': length_cm,
          'type': 'wall'
      }

      self.predicted_plan['walls'].append(wall)

    def convert_pixels_to_cm(self, pixel_length):
      cm_per_pixel = 6.293  
      return pixel_length * cm_per_pixel

    def is_overlapping(self, wall_start_1, wall_end_1, wall_start_2, wall_end_2, tolerance=50):

        # Check if both walls are vertically aligned
        if abs(wall_start_1[0] - wall_start_2[0]) <= tolerance and abs(wall_end_1[0] - wall_end_2[0]) <= tolerance:
            # Vertical walls with overlapping start and end points
            overlap_vertically = (max(wall_start_1[1], wall_end_1[1]) >= min(wall_start_2[1], wall_end_2[1])) and \
                                (min(wall_start_1[1], wall_end_1[1]) <= max(wall_start_2[1], wall_end_2[1]))
            if overlap_vertically:
                return True

        # Check if both walls are horizontally aligned
        if abs(wall_start_1[1] - wall_start_2[1]) <= tolerance and abs(wall_end_1[1] - wall_end_2[1]) <= tolerance:
            # Horizontal walls with overlapping start and end points
            overlap_horizontally = (max(wall_start_1[0], wall_end_1[0]) >= min(wall_start_2[0], wall_end_2[0])) and \
                                (min(wall_start_1[0], wall_end_1[0]) <= max(wall_start_2[0], wall_end_2[0]))
            if overlap_horizontally:
                return True

        return False

    def is_matching_column(self, placed_column, actual_column):
        MATCHING_TOLERANCE = 175  

        x_diff = abs(placed_column['x'] - actual_column['start_point'][0])
        y_diff = abs(placed_column['y'] - actual_column['start_point'][1])

        return x_diff <= MATCHING_TOLERANCE and y_diff <= MATCHING_TOLERANCE
    
    def is_matching_wall(self, placed_wall, actual_wall, tolerance=100):
        # Compare start points
        start_x_diff = abs(placed_wall['start']['start_point'][0] - actual_wall['start_point'][0])
        start_y_diff = abs(placed_wall['start']['start_point'][1] - actual_wall['start_point'][1])

        # Compare end points
        end_x_diff = abs(placed_wall['end']['end_point'][0] - actual_wall['end_point'][0])
        end_y_diff = abs(placed_wall['end']['end_point'][1] - actual_wall['end_point'][1])

        # Check if both start and end points are within the tolerance
        if (start_x_diff <= tolerance and start_y_diff <= tolerance and
            end_x_diff <= tolerance and end_y_diff <= tolerance):
            return True

        return False


def generate_foundation_plan(agent, new_floor_plan_features, reference_walls):
    env = FoundationDesignEnv(new_floor_plan_features, {'columns': [], 'walls': []}, reference_walls)
    obs, info = env.reset() 
    done = False
    
    while not done:
        action, _ = agent.predict(obs)  
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return {
        'columns': env.predicted_plan['columns'],
        'walls': env.predicted_plan['walls'] 
    }

def visualize_foundation_plan(foundation_plan, image_width=2000, image_height=2000):
    plt.figure(figsize=(10, 8))
    
    # Plot columns as squares with dimensions (25x25 cm)
    for column in foundation_plan['columns']:
        start_x, start_y = column['start_point']
        width = column['length_cm']  # Assuming column's width is equal to 'length_cm'
        height = column['height_cm']
        rect = plt.Rectangle((start_x, start_y), width, height, linewidth=1, edgecolor='blue', facecolor='blue')
        plt.gca().add_patch(rect)

    # Plot walls
    for wall in foundation_plan['walls']:
        plt.plot([wall['start_point'][0], wall['end_point'][0]], 
                 [wall['start_point'][1], wall['end_point'][1]], 
                 color='red', linewidth=2)

    # Annotate column start points with coordinates
    for column in foundation_plan['columns']:
        start_x, start_y = column['start_point']
        plt.annotate(f'({start_x:.0f}, {start_y:.0f})', (start_x, start_y), 
                     textcoords="offset points", xytext=(0, 10), ha='center')

    # Add plot details
    plt.title('Generated Foundation Plan')
    plt.xlabel('X Coordinates')
    plt.ylabel('Y Coordinates')
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.gca().invert_yaxis()  # Invert Y-axis to match the typical floor plan view
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.legend(['Wall', 'Column (Squares)'])
    plt.show()


if __name__ == "__main__":
        
    # Main execution flow
    floor_folder = 'path/to/floor/folder'
    floor_plan_features, reference_walls = preprocess_floor_data(floor_folder)

    floor_plan_features, foundation_plans, reference_walls = preprocess_floor_data(floor_folder)
    print(f"Total floor plans processed: {len(floor_plan_features)}")

    floor_data = load_json_files(floor_folder)

    env = FoundationDesignEnv(floor_plan_features[0], foundation_plans[0], reference_walls[0])
    model = PPO("MlpPolicy", env, verbose=1)

    num_epochs = 5
    total_timesteps_per_epoch = 10000 

    for epoch in range(num_epochs):
        for index in range(len(floor_plan_features)):
            floor_filename = floor_data[index][0]  
            foundation_filename = [index][0]  
            floor_plan = floor_plan_features[index] 
            foundation_plan = foundation_plans[index]  
            reference_wall = reference_walls[index]

            print(f"Processing Floor Plan: {floor_filename}, Foundation Plan: {foundation_filename}")  
            print(f"Features for current plan: {len(floor_plan)}")  

            env = FoundationDesignEnv(floor_plan, foundation_plan, reference_wall)
            model.set_env(env)  

            obs, info = env.reset() 
            print("Environment reset.")  

            model.learn(total_timesteps=total_timesteps_per_epoch)

    model.save("final_foundation_design_agent")

    generated_plan = generate_foundation_plan(model, floor_plan_features[0], reference_walls[0])
    print(f"Generated Foundation Plan: {generated_plan}")
    visualize_foundation_plan(generated_plan)
