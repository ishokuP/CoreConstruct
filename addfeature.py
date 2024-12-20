import json

def update_json_features(json_file_path, soil_type, building_type, num_storey, material_spec):
    # Mapping values to numbers
    soil_type_map = {
        "clay": 1,
        "silt": 2,
        "loam": 3,
        "sand": 4
    }

    building_type_map = {
        "residential": 1,
        "commercial": 2
    }

    num_storey_map = {
        "1storey": 1,
        "2storey": 2
    }

    material_spec_map = {
        "steel": 1,
        "wood-lm": 2,
        "rc": 3
    }


    # Convert form values to numbers using the maps
    soil_type_value = soil_type_map.get(soil_type, 0)  # Default to 0 if not found
    building_type_value = building_type_map.get(building_type, 0)
    num_storey_value = num_storey_map.get(num_storey, 0)
    material_spec_value = material_spec_map.get(material_spec, 0)

    # Load the existing JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Check if 'features' exists in the JSON; if not, create it
    if 'features' not in data:
        data['features'] = {}
    
    # Append the new features based on the form data
    data['features']['canvas_width'] = data['features'].get('canvas_width', 0)
    data['features']['canvas_height'] = data['features'].get('canvas_height', 0)
    data['features']['soil_type'] = soil_type_value
    data['features']['building_type'] = building_type_value
    data['features']['num_storey'] = num_storey_value
    data['features']['material_spec'] = material_spec_value

    # Write the updated JSON back to the file
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("JSON features updated successfully!")



def transfer_canvas_dimensions(source_json_path, destination_json_path):
    """
    Transfers the canvas width and height from the source JSON file to the destination JSON file.
    """
    # Load the source JSON data
    with open(source_json_path, 'r') as source_file:
        source_data = json.load(source_file)
    
    # Extract canvas dimensions from the source file
    canvas_width = source_data.get('features', {}).get('canvas_width', 0)
    canvas_height = source_data.get('features', {}).get('canvas_height', 0)

    # Load the destination JSON data
    with open(destination_json_path, 'r') as destination_file:
        destination_data = json.load(destination_file)

    # Update or create the 'features' section in the destination JSON
    if 'features' not in destination_data:
        destination_data['features'] = {}
    
    destination_data['features']['canvas_width'] = canvas_width
    destination_data['features']['canvas_height'] = canvas_height

    # Write the updated JSON back to the destination file
    with open(destination_json_path, 'w') as destination_file:
        json.dump(destination_data, destination_file, indent=4)
    print("Canvas dimensions transferred successfully!")


# Example usage
if __name__ == "__main__":
    json_file_path = 'output/yolov8/output.json'  # Path to your JSON file

    # Sample data, this should come from your Flask app
    soil_type = "Clay"
    building_type = "Residential"
    num_storey = "2-storey building"
    material_spec = "Reinforced Steel"
    
    update_json_features(json_file_path, soil_type, building_type, num_storey, material_spec)
