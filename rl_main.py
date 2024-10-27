import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import os
import matplotlib.pyplot as plt
import random

def load_json_files(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_data = []

    for json_file in json_files:
        with open(os.path.join(folder_path, json_file)) as f:
            data = json.load(f)
            json_data.append((json_file, data))

    return json_data

def preprocess_all_data(floor_folder, foundation_folder):
    floor_data = load_json_files(floor_folder)
    foundation_data = load_json_files(foundation_folder)

    # Extract the filenames for comparison
    floor_filenames = set([f[0].lower() for f in floor_data])
    foundation_filenames = set([f[0].lower() for f in foundation_data])

    # Find the common files in both directories
    common_filenames = floor_filenames.intersection(foundation_filenames)

    if not common_filenames:
        print("No common filenames found between floor and foundation folders.")
        return [], [], []

    # Now we need to process only the common files
    floor_plan_features = []
    foundation_plans = []
    reference_walls = []

    for (floor_filename, floor_plan), (foundation_filename, foundation_plan) in zip(floor_data, foundation_data):
        # Convert filenames to lowercase for case-insensitive matching
        if floor_filename.lower() in common_filenames and foundation_filename.lower() in common_filenames:
            # Extract walls, midpoints, and corners
            midpoints = floor_plan['averaged_midpoints']
            corners = floor_plan['averaged_corners']
            walls = floor_plan['walls']

            features = [{'x': point['x'], 'y': point['y'], 'type': 'midpoint'} for point in midpoints] + \
                       [{'x': point['x'], 'y': point['y'], 'type': 'corner'} for point in corners]

            # Prepare foundation details
            foundation_columns = foundation_plan['columns']
            foundation_walls = foundation_plan['walls']

            floor_plan_features.append(features)
            reference_walls.append(walls)
            foundation_plans.append({'columns': foundation_columns, 'walls': foundation_walls})
        else:
            print(f"Skipping due to mismatch: {floor_filename}, {foundation_filename}")

    return floor_plan_features, foundation_plans, reference_walls

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
        self.proximity_tolerance=50
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_features,), dtype=np.float32)

    def step(self, action):
      reward = 0
      done = False

      # Step 1: Place columns as before based on the action
      if self.current_step < len(self.floor_plan_features):
          feature = self.floor_plan_features[self.current_step]

          if action == 0:  # Place column based on the current feature
              if feature['type'] == 'corner':
                  self.place_column(feature)
                  reward += 1
              elif feature['type'] == 'midpoint':
                  if self.is_valid_column_point(feature):
                      self.place_column(feature)
                      reward += 0.5

          elif action == 1:  # Try placing additional columns along reference walls
              additional_columns = self.find_additional_column_points(feature)
              for col in additional_columns:
                  if self.is_valid_column_point(col) and not self.is_near_existing_column(col):
                      self.place_column(col)
                      reward += 0.5

          '''column_count = len(self.predicted_plan['columns'])
          reference_wall_count = len(self.reference_walls)
          if column_count >= 2:
              for i in range(len(self.predicted_plan['columns']) - 1):
                  for j in range(i + 1, len(self.predicted_plan['columns'])):
                      col1 = self.predicted_plan['columns'][i]
                      col2 = self.predicted_plan['columns'][j]

                      if self.can_connect_columns(col1, col2):
                          matching_wall = any(
                              self.is_matching_wall({'start': col1, 'end': col2}, wall)
                              for wall in self.reference_walls
                          )
                          if matching_wall:
                            self.place_wall(col1, col2)
                          else:
                              # Fallback logic: if fewer walls placed than reference, add more walls
                              if len(self.predicted_plan['walls']) < reference_wall_count:
                                  self.place_wall(col1, col2)'''

          self.current_step += 1
          done = self.current_step >= len(self.floor_plan_features)
          self.snap_and_place_walls(tolerance=50)
          #self.place_all_possible_walls()

          # If predicted walls exceed reference walls, remove extra walls
          #if len(self.predicted_plan['walls']) > len(self.reference_walls):
           #   self.remove_non_matching_walls()

      # Step 2: Try to place walls by matching reference walls to predicted columns
      '''snapped_walls = []
      for ref_wall in self.reference_walls:
          # Check if matching column pair exists
          start_col, end_col = self.matching_column_pair(ref_wall)

          # If both start and end columns are found and there's no overlap
          if start_col is not None and end_col is not None:
            if not self.has_overlap({'start_point': start_col['center_point'],
                                                              'end_point': end_col['center_point']}):
                # Snap wall to center points of the matched columns
                snapped_wall = {
                    'start_point': start_col['center_point'],
                    'end_point': end_col['center_point'],
                    'length_cm': self.convert_pixels_to_cm(
                        np.linalg.norm(np.array(start_col['center_point']) - np.array(end_col['center_point']))
                    )
                }
                # Add the snapped wall to the predicted plan
                self.predicted_plan['walls'].append(snapped_wall)
                snapped_walls.append(snapped_wall)
                reward += 2  # Reward for correctly placing a snapped wall'''

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
        ALIGNMENT_TOLERANCE = 50
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
      column_diag = 12.5
      column_x = 12.5

      # Adjust the column representation to include start and end points, width, and height
      column = {
          'start_point': (col['x']- column_x, col['y']- column_diag),
          'end_point': (col['x'] + column_x, col['y'] + column_diag),
          'length_cm': column_width,  # Assuming width is equivalent to length for simplicity
          'height_cm': column_height,
          'center_point': (col['x'],col['y']),
          'type': 'column'
      }
      self.predicted_plan['columns'].append(column)

    def place_all_possible_walls(self, alignment_tolerance=50):
      column_count = len(self.predicted_plan['columns'])

      # Iterate over all unique column pairs
      for i in range(column_count - 1):
          for j in range(i + 1, column_count):
              col1 = self.predicted_plan['columns'][i]
              col2 = self.predicted_plan['columns'][j]

              # Define the start and end points of the wall
              wall_start = col1['center_point']
              wall_end = col2['center_point']

              # Check if the wall is approximately vertical or horizontal
              if abs(wall_start[0] - wall_end[0]) <= alignment_tolerance or abs(wall_start[1] - wall_end[1]) <= alignment_tolerance:
                  # Create a wall representation
                  wall = {
                      'start_point': wall_start,
                      'end_point': wall_end,
                      'length_cm': self.convert_pixels_to_cm(
                          np.sqrt((wall_end[0] - wall_start[0])**2 + (wall_end[1] - wall_start[1])**2)
                      ),
                      'type': 'wall'
                  }

                  # Check if the wall overlaps with any existing walls
                  if not self.has_overlap(wall):
                      # Append the wall to the predicted plan if no overlap is found
                      self.predicted_plan['walls'].append(wall)
    
    def remove_non_matching_walls(self):
      matched_walls = []

      # Iterate over the predicted walls
      for wall in self.predicted_plan['walls']:
          # Check if the wall matches any of the reference walls within the given tolerance
          match_found = any(
              self.is_matching_wall(wall, ref_wall)
              for ref_wall in self.reference_walls
          )

          if match_found:
              matched_walls.append(wall)

      # Update the predicted plan to include only matching walls
      self.predicted_plan['walls'] = matched_walls

    def place_wall(self, start_col, end_col):
      # Calculate the pixel length using the centers of the columns
      pixel_length = np.sqrt((end_col['center_point'][0] - start_col['center_point'][0]) ** 2 +
                            (end_col['center_point'][1] - start_col['center_point'][1]) ** 2)
      length_cm = self.convert_pixels_to_cm(pixel_length)

      # Determine wall orientation and place it accordingly based on column centers
      if abs(start_col['center_point'][0] - end_col['center_point'][0]) <= self.threshold:  # Vertical wall
          wall_start = (start_col['center_point'][0], min(start_col['center_point'][1], end_col['center_point'][1]))
          wall_end = (start_col['center_point'][0], max(start_col['center_point'][1], end_col['center_point'][1]))
      elif abs(start_col['center_point'][1] - end_col['center_point'][1]) <= self.threshold:  # Horizontal wall
          wall_start = (min(start_col['center_point'][0], end_col['center_point'][0]), start_col['center_point'][1])
          wall_end = (max(start_col['center_point'][0], end_col['center_point'][0]), start_col['center_point'][1])
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

    def is_overlapping(self, start1, end1, start2, end2):
        if start1[0] == end1[0] and start2[0] == end2[0]:
            return start1[0] == start2[0] and not (end1[1] < start2[1] or end2[1] < start1[1])
        elif start1[1] == end1[1] and start2[1] == end2[1]:
            return start1[1] == start2[1] and not (end1[0] < start2[0] or end2[0] < start1[0])
        return False

    def is_too_close_to_existing_columns(self, new_column):
        for existing_col in self.predicted_plan['columns']:
            distance = np.sqrt(
                (new_column['x'] - existing_col['center_point'][0]) ** 2 +
                (new_column['y'] - existing_col['center_point'][1]) ** 2
            )
            if distance <= self.proximity_tolerance:
                return True  # Too close to an existing column
        return False

    def is_matching_wall(self, placed_wall, actual_wall, tolerance=50):
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

    def find_additional_column_points(self, feature):
          additional_points = []
          for wall in self.reference_walls:
              start_x, start_y = wall['start_point']
              end_x, end_y = wall['end_point']

              # Determine potential points for additional columns along the wall
              num_points = 15  # Number of points to consider along each wall
              for i in range(1, num_points):
                  x = start_x + i * (end_x - start_x) / num_points
                  y = start_y + i * (end_y - start_y) / num_points

                    # Check if the point aligns with the reference plan
                  if self.is_valid_column_point({'x': x, 'y': y}):
                      additional_points.append({'x': x, 'y': y})

          return additional_points

    def is_valid_column_point(self, point):
            MATCHING_TOLERANCE = 50  # Adjust as needed for valid placement
            for col in self.reference_plan['columns']:
                if abs(point['x'] - col['start_point'][0]) <= MATCHING_TOLERANCE and \
                abs(point['y'] - col['start_point'][1]) <= MATCHING_TOLERANCE:
                    return True
            return False

    def is_far_from_existing_columns(self, feature, min_distance=700):
      for col in self.predicted_plan['columns']:
          # Calculate Euclidean distance from existing column center
          distance = np.sqrt(
              (feature['x'] - col['center_point'][0])**2 +
              (feature['y'] - col['center_point'][1])**2
          )
          if distance < min_distance:
              return False  # Too close to an existing column
      return True  # Sufficiently far from existing columns
    def snap_to_column_centers(self, feature, tolerance=40):
      snapped_feature = feature.copy()
      start_snapped = False
      end_snapped = False
      column_count = len(self.predicted_plan['columns'])

      # Iterate over all unique column pairs
      for i in range(column_count - 1):
          for j in range(i + 1, column_count):
              col1 = self.predicted_plan['columns'][i]
              col2 = self.predicted_plan['columns'][j]

              # Check if the start point matches the first column within tolerance
              if (
                  abs(feature['start_point'][0] - col1['center_point'][0]) <= tolerance and
                  abs(feature['start_point'][1] - col1['center_point'][1]) <= tolerance
              ):
                  snapped_feature['start_point'] = col1['center_point']
                  start_snapped = True

              # Check if the end point matches the second column within tolerance
              if (
                  abs(feature['end_point'][0] - col2['center_point'][0]) <= tolerance and
                  abs(feature['end_point'][1] - col2['center_point'][1]) <= tolerance
              ):
                  snapped_feature['end_point'] = col2['center_point']
                  end_snapped = True

              # Break out of the loop if both start and end are snapped
              if start_snapped and end_snapped:
                  break

      # Check for overlaps before returning the snapped feature
      if self.has_overlap(snapped_feature):
          return feature, False, False  # Return original feature if there's an overlap

      return snapped_feature, start_snapped, end_snapped

    def has_overlap(self, new_wall):
      for existing_wall in self.predicted_plan['walls']:
          if self.is_overlapping(new_wall['start_point'], new_wall['end_point'],
                                existing_wall['start_point'], existing_wall['end_point']):
              return True  # Overlap detected
      return False  # No overlap
    def matching_column_pair(self, ref_wall, tolerance=50):
      start_col = None
      end_col = None

      # Iterate through all column pairs in the predicted plan
      for i in range(len(self.predicted_plan['columns']) - 1):
          for j in range(i + 1, len(self.predicted_plan['columns'])):
              col1 = self.predicted_plan['columns'][i]
              col2 = self.predicted_plan['columns'][j]

              # Check if start point of ref_wall matches col1's center
              if (abs(ref_wall['start_point'][0] - col1['center_point'][0]) <= tolerance and
                      abs(ref_wall['start_point'][1] - col1['center_point'][1]) <= tolerance):
                  start_col = col1

              # Check if end point of ref_wall matches col2's center
              if (abs(ref_wall['end_point'][0] - col2['center_point'][0]) <= tolerance and
                      abs(ref_wall['end_point'][1] - col2['center_point'][1]) <= tolerance):
                  end_col = col2

              # If both start and end columns are matched, return them
              if start_col and end_col:
                  return start_col, end_col

      return None, None

    def is_near_existing_column(self, column, tolerance=50):
      # Direct comparison of the new column's coordinates with existing column centers
      for existing_col in self.predicted_plan['columns']:
          if (abs(column['x'] - existing_col['center_point'][0]) <= tolerance and
                  abs(column['y'] - existing_col['center_point'][1]) <= tolerance):
              return True  # New column is near an existing one
      return False  # No nearby column found
    def connect_column_pairs(self, tolerance=30):
      column_count = len(self.predicted_plan['columns'])

      # Iterate over all unique column pairs
      for i in range(column_count - 1):
          for j in range(i + 1, column_count):
              col1 = self.predicted_plan['columns'][i]
              col2 = self.predicted_plan['columns'][j]

              # Define the start and end points of the wall as the center points of the columns
              wall_start = col1['center_point']
              wall_end = col2['center_point']

              # Create a wall representation
              wall = {
                  'start_point': wall_start,
                  'end_point': wall_end,
                  'length_cm': self.convert_pixels_to_cm(
                      np.sqrt((wall_end[0] - wall_start[0])**2 + (wall_end[1] - wall_start[1])**2)
                  ),
                  'type': 'wall'
              }

              # Check if the wall overlaps with any existing walls
              if not self.has_overlap(wall):
                  # Append the wall to the predicted plan if no overlap is found
                  self.predicted_plan['walls'].append(wall)
                  print(f"Added wall from {wall_start} to {wall_end}")
              else:
                  print(f"Skipped wall from {wall_start} to {wall_end} due to overlap")
    
    def snap_and_place_walls(self, tolerance=50):
    # Iterate through each reference wall
      for ref_wall in self.reference_walls:
          start_point = ref_wall['start_point']
          end_point = ref_wall['end_point']

          # Find the closest column to the start point
          closest_start_col = self.get_closest_column(start_point, tolerance)
          # Find the closest column to the end point
          closest_end_col = self.get_closest_column(end_point, tolerance)

          # If both start and end columns are found, proceed to snapping
          if closest_start_col is not None and closest_end_col is not None:
              snapped_wall = {
                  'start_point': closest_start_col['center_point'],
                  'end_point': closest_end_col['center_point'],
                  'length_cm': self.convert_pixels_to_cm(
                      np.linalg.norm(
                          np.array(closest_start_col['center_point']) - 
                          np.array(closest_end_col['center_point'])
                      )
                  ),
                  'type': 'wall'
              }

              # Check for overlaps before adding the snapped wall
              if not self.has_overlap(snapped_wall):
                  self.predicted_plan['walls'].append(snapped_wall)

    def get_closest_column(self, point, tolerance=50):
        closest_col = None
        min_distance = float('inf')

        # Iterate over all columns in the predicted plan
        for col in self.predicted_plan['columns']:
            col_center = col['center_point']
            distance = np.linalg.norm(np.array(point) - np.array(col_center))

            # Check if the column is within the tolerance and is closer than the current minimum
            if distance <= tolerance and distance < min_distance:
                closest_col = col
                min_distance = distance

        return closest_col
    def process_reference_walls(self, tolerance= 50):
      # Iterate through each reference wall
      for ref_wall in self.reference_walls:
          start_point = ref_wall['start_point']
          end_point = ref_wall['end_point']
          snapped_start, snapped_end = None, None
          start_snapped, end_snapped = False, False

          # Step 1: Find potential column pairings for the reference wall
          column_count = len(self.predicted_plan['columns'])
          for i in range(column_count - 1):
              for j in range(i + 1, column_count):
                  col1 = self.predicted_plan['columns'][i]
                  col2 = self.predicted_plan['columns'][j]

                  # Check if the start point matches col1 within tolerance
                  if (
                      abs(start_point[0] - col1['center_point'][0]) <= tolerance and
                      abs(start_point[1] - col1['center_point'][1]) <= tolerance
                  ):
                      snapped_start = col1['center_point']
                      start_snapped = True

                  # Check if the end point matches col2 within tolerance
                  if (
                      abs(end_point[0] - col2['center_point'][0]) <= tolerance and
                      abs(end_point[1] - col2['center_point'][1]) <= tolerance
                  ):
                      snapped_end = col2['center_point']
                      end_snapped = True

                  # If both start and end are snapped, break the loop
                  if start_snapped and end_snapped:
                      break
              if start_snapped and end_snapped:
                  break

          # If both start and end are snapped, proceed to the next step
          if start_snapped and end_snapped:
              snapped_wall = {'start_point': snapped_start, 'end_point': snapped_end}

              # Step 2: Check for overlap with existing walls
              if not self.has_overlap(snapped_wall):
                  # Step 3: Append to predicted plan
                  self.predicted_plan['walls'].append(snapped_wall)
              else:
                  print(f"Overlap detected. Wall skipped: Start {snapped_start}, End {snapped_end}")
          else:
              print(f"No matching columns found for wall: Start {start_point}, End {end_point}")
        
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


def run():
    floor_folder = 'Floor'
    foundation_folder = 'Foundation'

    # Preprocess all data and ensure correct pairing
    floor_plan_features, foundation_plans, reference_walls = preprocess_all_data(floor_folder, foundation_folder)
    print(f"Total floor plans processed: {len(floor_plan_features)}")

    floor_data = load_json_files(floor_folder)
    foundation_data = load_json_files(foundation_folder)

    # Initialize the environment with the first matched plans
    env = FoundationDesignEnv(floor_plan_features[0], foundation_plans[0], reference_walls[0])
    model = PPO("MlpPolicy", env, verbose=1)

    num_epochs = 5
    total_timesteps_per_epoch = 10000

    # Use preprocessed, correctly paired data for training
    for epoch in range(num_epochs):
        for index in range(len(floor_plan_features)):
            floor_filename = floor_data[index][0]
            foundation_filename = foundation_data[index][0]
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

    # Save the trained model
    model.save("/content/drive/MyDrive/Dataset/Training/final_foundation_design_agent_finalnafinal")
    model.save("final_foundation_design_agent_final")

    # Test the generated plan
    generated_plan = generate_foundation_plan(model, floor_plan_features[0], reference_walls[0])
    print(f"Generated Foundation Plan: {generated_plan}")
    visualize_foundation_plan(generated_plan)
