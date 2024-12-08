import math
import json

# Function to calculate dead load
def calculate_dead_load(json_file,wind_speed=180,roof_type="Flat", wall_height=2.9, wall_thickness=0.2, wall_density=2400, floor_density=2400):
    """
    Returns:
    - tuple: Total wall load (kN), floor load (kN), roof load (kN), wall wind load (kN), roof wind load (kN).
    """
    # Set Cp values based on roof type
    if roof_type == "flat":
        Cp_wall = 0.8
        Cp_roof = -0.6
    elif roof_type == "sloped":
        Cp_wall = 0.8
        Cp_roof = -0.3
    else:
        raise ValueError("Invalid roof type. Only 'Flat' and 'Sloped' are allowed.")

    # Parse the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Calculate total wall load
    walls = data.get("walls", [])
    total_wall_length = sum(wall['length_cm'] / 1000 for wall in walls)  # Convert cm to meters
    total_wall_load = total_wall_length * wall_height * wall_thickness * wall_density

    total_wall_load_kN = total_wall_load * 0.00980665  # Convert from kg to kN

    # Calculate wall area for wind load
    wall_area = total_wall_length * wall_height

    # Calculate floor/roof area from JSON
    floor_area = calculate_polygon_area_from_json(json_file)
    roof_area = floor_area  # Assume roof area equals floor area

    # Floor load
    floor_load = floor_area * 0.125 * floor_density
    floor_load_kN = floor_load * 0.00980665  # Convert from kg to kN

    # Roof load
    roof_load = calculate_roof_load(roof_type, roof_area, material_density=7850)
    roof_load_kN = roof_load * 0.00980665 # Convert from kg to kN

    # Wind load
    wall_wind_load, roof_wind_load = calculate_wind_load(wind_speed, wall_area, roof_area, Cp_wall, Cp_roof)

    # Convert wind load to kN (from kg)
    wall_wind_load_kN = wall_wind_load * 0.00980665  # Convert from kg to kN
    roof_wind_load_kN = roof_wind_load * 0.00980665  # Convert from kg to kN

    return total_wall_load_kN, floor_load_kN, roof_load_kN, wall_wind_load_kN, roof_wind_load_kN

# Function to calculate roof load
def calculate_roof_load(roof_type, area, material_density):
    if roof_type == "flat":
        roof_load = area * 0.01 * material_density
    else:
        raise ValueError("Unknown roof type")
    
    return roof_load

# Function to calculate live load
def calculate_live_load(floor_area):
    return floor_area * 1.5

# Function to calculate wind load
def calculate_wind_load(wind_speed, wall_area, roof_area, Cp_wall, Cp_roof):

    q = 0.00256 * wind_speed**2 * 0.85 * 0.85
    wall_load = q * wall_area * Cp_wall
    roof_load = q * roof_area * Cp_roof

    return wall_load, roof_load

# Function to calculate seismic load
def calculate_seismic_load(total_weight, seismic_coefficient):
    return total_weight * seismic_coefficient

def order_vertices(corners):
    # Find the centroid
    n = len(corners)
    centroid_x = sum(corner['x'] for corner in corners) / n
    centroid_y = sum(corner['y'] for corner in corners) / n

    # Calculate angles and sort vertices
    corners_with_angles = [
        {**corner, 'angle': math.atan2(corner['y'] - centroid_y, corner['x'] - centroid_x)} for corner in corners
    ]
    corners_with_angles.sort(key=lambda corner: corner['angle'])  # Sort by angle

    # Return only the ordered vertices
    return [{'x': corner['x'], 'y': corner['y']} for corner in corners_with_angles]


# Function to calculate the polygon area from the JSON file (floor/roof area)
def calculate_polygon_area_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        corners = data.get("averaged_corners", [])

    if not corners:
        raise ValueError("No averaged_corners found in the JSON file.")

    ordered_corners = order_vertices(corners)
    corners = ordered_corners
    n = len(corners)
    area = 0
    for i in range(n):
        x1, y1 = corners[i]['x'], corners[i]['y']
        x2, y2 = corners[(i + 1) % n]['x'], corners[(i + 1) % n]['y']  # Next vertex (cyclic)
        area += x1 * y2 - y1 * x2

    area = abs(area) / 2.0
    real_area = (area * 44.7436)/1000000
    print(f"house area: {real_area}")
    return real_area  # Convert cm² to m²

# Function to calculate foundation load (footing size)
def calculate_foundation_load(json_data, soil_type):
    soil_bearing_capacities = {
        "Clay": 125,   # kN/m²
        "Loam": 125,   # kN/m²
        "Sand": 175,   # kN/m²
        "Silt": 100    # kN/m²
    }
    FOOTING_SIZE = 0.00746  # in meters (850mm x 850mm footings)
    SOIL_BEARING_CAPACITY = soil_bearing_capacities[soil_type]  # in kN/m²
    
    # Calculate footing area
    footing_area = FOOTING_SIZE * FOOTING_SIZE  # in m²
    
    # Calculate the load capacity for each footing
    load_per_footing = SOIL_BEARING_CAPACITY * footing_area  # in kN
    
    # Parse the JSON data to count the number of columns
    columns = json_data.get('columns', [])
    num_columns = len(columns)
    
    # Calculate the total load
    total_load = num_columns * load_per_footing
    
    return total_load

# Main function to calculate total load (deadload + wind load + liveload + seismic load)
def mainFunction(json_fileRL,json_fileCNN,location,roof_type,valueMeter):
    seismic_zone, wind_speed, soil_type = (get_location_constants(location).values())
    if seismic_zone == 4:
        seismic_coefficient = 0.3
    else:
        seismic_coefficient = 0.15
    # Calculate Dead Load
    total_wall_load, floor_load, roof_load, wall_wind_load, roof_wind_load = calculate_dead_load(json_fileCNN,wind_speed,roof_type)
    
    # Calculate Live Load
    floor_area = calculate_polygon_area_from_json(json_fileCNN)
    live_load = calculate_live_load(floor_area)
    
    # Average Wind Load (assuming equal wind distribution across all walls)
    walls = json.loads(open(json_fileCNN).read()).get('walls', [])
    average_wall_wind_load = sum([wall_wind_load for _ in walls])/4
    
    # Calculate Seismic Load
    total_weight = total_wall_load + floor_load + roof_load + live_load
    seismic_load = calculate_seismic_load(total_weight, seismic_coefficient)
    foundation_load = calculate_foundation_load(json_fileRL,soil_type,valueMeter)
    
    # Total Load Calculation
    total_building_load = total_wall_load + floor_load + roof_load + live_load + average_wall_wind_load + seismic_load + roof_wind_load
    
    dead_load = total_wall_load + floor_load + roof_load
    
    # Output the results
    print(f"Total Wall Load: {total_wall_load:.2f} kN")
    print(f"Total Floor Load: {floor_load:.2f} kN")
    print(f"Total Roof Load: {roof_load:.2f} kN")
    print(f"Total Dead Load: {dead_load:.2f} kN")
    print(f"Average Wind Load: {average_wall_wind_load + roof_wind_load:.2f} kN")
    print(f"Total Live Load: {live_load:.2f} kN")
    print(f"Seismic Load: {seismic_load:.2f} kN")
    print(f"Total Building Load: {total_building_load:.2f} kN")
    print(f"Total Foundation Load: {foundation_load:.2f} kN")
    
    return dead_load,total_wall_load,floor_load,roof_load,live_load,average_wall_wind_load,seismic_load,total_building_load,foundation_load

def get_location_constants(location, soil_type=None):
    # Define mappings
    location_data = {
        "metroManila": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Clay"},
        "Cebu": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Loam"},
        "Palawan": {"seismic_zone": 2, "wind_speed": 144, "default_soil": "Sand"},
        "Davao": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Silt"},
        "Batanes": {"seismic_zone": 4, "wind_speed": 198, "default_soil": "Sand"},
        "Iloilo": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Loam"},
        "Leyte": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Clay"},
        "Zamboanga": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Silt"},
        "Pangasinan": {"seismic_zone": 2, "wind_speed": 144, "default_soil": "Sand"},
        "Cagayan": {"seismic_zone": 2, "wind_speed": 144, "default_soil": "Silt"},
        "negrosOccidental": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Loam"},
        "Bukidnon": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Clay"},
        "Surigao": {"seismic_zone": 4, "wind_speed": 198, "default_soil": "Silt"},
        "ilocosNorte": {"seismic_zone": 2, "wind_speed": 144, "default_soil": "Sand"},
        "misamisOriental": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Loam"},
        "Benguet": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Silt"},
        "Batangas": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Clay"},
        "Quezon": {"seismic_zone": 4, "wind_speed": 162, "default_soil": "Loam"},
        "Albay": {"seismic_zone": 4, "wind_speed": 180, "default_soil": "Clay"}
    }

    # Get location-specific data
    data = location_data.get(location, {"seismic_zone": 2, "wind_speed": 144, "default_soil": "Loam"})
    seismic_zone = data["seismic_zone"]
    wind_speed = data["wind_speed"]
    default_soil = data["default_soil"]

    # Use user-defined soil type or default
    soil = soil_type if soil_type else default_soil
    
    print(f"Wind Speed: {wind_speed}")
    print(f"Seismic Zone: {seismic_zone}")
    print(f"Soil Type: {soil}")
    

    return {
        "seismic_zone": seismic_zone,
        "wind_speed": wind_speed,
        "soil_type": soil
    }


def calculate_foundation_load(file_path,soil_type,valueMeter):
    with open(file_path, 'r') as f:
        json_data = json.load(f)  # Load the JSON data from the file
    # Constants (these can be adjusted dynamically if needed)
    soil_bearing_capacities = {
        "Clay": 125,   # kN/m²
        "Loam": 125,   # kN/m²
        "Sand": 175,   # kN/m²
        "Silt": 100    # kN/m²
    }
    FOOTING_SIZE = valueMeter  # in meters (850mm x 850mm footings)
    SOIL_BEARING_CAPACITY = soil_bearing_capacities[soil_type]  # in kN/m²
    
    # Calculate footing area
    footing_area = FOOTING_SIZE * FOOTING_SIZE  # in m²
    
    # Calculate the load capacity for each footing
    load_per_footing = SOIL_BEARING_CAPACITY * footing_area  # in kN
    
    # Parse the JSON data to count the number of columns
    columns = json_data.get('columns', [])
    num_columns = len(columns)
    
    # Calculate the total load
    total_load = num_columns * load_per_footing
    
    return total_load


#main('FloorplanPrediction.json', soil_type="Loam", seismic_coefficient=0.3)
