import json
import math
import os 
# Conversion factor for pixels to centimeters
px_to_cm = 6.923

def transform_predictions(input_data):
    """
    Transforms input predictions JSON into the desired format with start and end points, 
    wall length in centimeters, averaged corners, and midpoints.
    """
    predictions = input_data.get('predictions', [])
    image_width = int(input_data['image']['width'])
    image_height = int(input_data['image']['height'])
    
    output_data = {
        "file_name": os.path.basename(predictions[0]['image_path']) if predictions else "",
        "walls": [],
        "averaged_corners": [],
        "averaged_midpoints": [],
        "features": {
            "canvas_width": image_width,
            "canvas_height": image_height
        }
    }

    # Helper lists for averaging corners and midpoints
    all_corners = []
    all_midpoints = []

    for prediction in predictions:
        # Calculate start and end points of each wall
        x, y = prediction['x'], prediction['y']
        width, height = prediction['width'], prediction['height']

        # Determine orientation: horizontal or vertical wall
        if width > height:  # Horizontal wall
            start_point = [int(x - width / 2), int(y)]
            end_point = [int(x + width / 2), int(y)]
        else:  # Vertical wall
            start_point = [int(x), int(y - height / 2)]
            end_point = [int(x), int(y + height / 2)]

        # Calculate wall length in pixels and convert to cm
        length_px = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        length_cm = length_px * px_to_cm

        # Append wall data to output
        wall_data = {
            "start_point": start_point,
            "end_point": end_point,
            "length_cm": round(length_cm, 2)
        }
        output_data["walls"].append(wall_data)

        # Collect corners and midpoints for averaging
        all_corners.extend([start_point, end_point])
        midpoint = [(start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2]
        all_midpoints.append(midpoint)

    # Average the corners
    averaged_corners = average_points(all_corners)
    averaged_midpoints = average_points(all_midpoints)

    output_data["averaged_corners"] = [{"x": corner[0], "y": corner[1]} for corner in averaged_corners]
    output_data["averaged_midpoints"] = [{"x": midpoint[0], "y": midpoint[1]} for midpoint in averaged_midpoints]

    return output_data


def average_points(points, radius=150):
    """
    Averages nearby points to reduce noise in corners or midpoints.
    """
    averaged_points = []
    used = [False] * len(points)

    for i, point in enumerate(points):
        if used[i]:
            continue

        close_points = [point]
        for j, other_point in enumerate(points):
            if i != j and not used[j]:
                dist = math.sqrt((point[0] - other_point[0]) ** 2 + (point[1] - other_point[1]) ** 2)
                if dist < radius:
                    close_points.append(other_point)
                    used[j] = True

        # Calculate the average of close points
        avg_x = sum(p[0] for p in close_points) // len(close_points)
        avg_y = sum(p[1] for p in close_points) // len(close_points)
        averaged_points.append((avg_x, avg_y))
        used[i] = True

    return averaged_points


if __name__ == "__main__":
    # Path to input JSON file
    input_file_path = 'inference_results.json'  # Replace with your actual input JSON file path

    # Load input data from JSON file
    try:
        with open(input_file_path, 'r') as json_file:
            input_data = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{input_file_path}'.")
        exit(1)

    # Transform the input data
    output_data = transform_predictions(input_data)

    # Save the transformed output to a JSON file
    output_file_path = 'transformed_output.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Transformed output saved as '{output_file_path}'")