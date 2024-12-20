from flask import Flask, render_template, request, jsonify, flash, redirect, url_for,send_from_directory
import os
from PIL import Image
import json
import numpy as np
from werkzeug.utils import secure_filename
import shutil



# Importing coreconstruct features

from floorplantojson import process_image
from jsonFormatter import transform_predictions
from addfeature import transfer_canvas_dimensions
from foundationJSONtoimage import generateFoundationPlan
from ANN import test_model
from RL import rl_module
from load_calculation2 import get_location_constants


# Define the mappings
soil_type_map = {
    "Clay": 1,
    "Silt": 2,
    "Loam": 3,
    "Sand": 4
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

Barsize_map = {
    "10mm": 10,
    "12mm": 12,
    "16mm": 16,
    "20mm": 20,
}

slope_map = {
    "fourOverTwelve": 18.43,
    "sixOverTwelve": 26.57,
    "eightOverTwelve": 33.69,
    "twelveOverTwelve": 45
}



app = Flask(__name__)
app.secret_key = 'wowCoreandConstruct'

# Folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
OUTPUT_DIR = 'output/'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_DIR'] = OUTPUT_DIR


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Root route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_generate', methods=['POST'])
def analyze_generate():
    print("Form submitted")  # Indicate that the form has been submitted

    # Collect the form data
    uploaded_file = request.files.get('floor-plan')
    building_type = request.form.get('buildingType')
    num_storey = request.form.get('numStorey')
    material_spec = request.form.get('materialSpecs')
    barSize = request.form.get('barSize')
    location = request.form.get('location')
    roofType = request.form.get('roofType')

    # Print the values received for debugging
    print(f"Uploaded File: {uploaded_file.filename if uploaded_file else 'No file uploaded'}")
    print(f"Building Type: {building_type}")
    print(f"Number of Storeys: {num_storey}")
    print(f"Material Spec: {material_spec}")
    print(f"Bar Size: {barSize}")
    print(f"Location: {location}")
    print(f"Roof Type: {roofType}")

    # Perform backend validation
    missing_fields = []

    missing_fields = [
        field for field, value in {
            "Floor Plan": uploaded_file,
            "Building Type": building_type,
            "Number of Storeys": num_storey,
            "Material Specifications": material_spec,
            "Bar Size": barSize,
            "Location": location,
            "Roof Type": roofType,
        }.items() if not value
    ]

    # If any fields are missing, flash a message and redirect
    if missing_fields:
        flash(f"Please fill in the following fields: {', '.join(missing_fields)}")
        return redirect('/')  # Redirect back to the main page

    # Check if the file is present and allowed
    if not allowed_file(uploaded_file.filename):
        flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF).')
        return redirect('/')

    #
    # Putting the file to the upload folder and keeping the original filename
    #

    # Sanitize the filename
    filename = secure_filename(uploaded_file.filename)

    # Create the uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Set the full file path with the original filename
    uploadPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the uploaded file with the original name
    uploaded_file.save(uploadPath)

    #
    # This is for the Floorplan to JSON Part
    #
    preCleanedJSON = os.path.join(
        app.config['OUTPUT_DIR'], 'yolov8', 'uncleanedJSON', 'preCleanedPredictions.json')
    maskOutput = os.path.join(
        app.config['OUTPUT_DIR'], 'yolov8', 'floorplanMask.png')
    outputImage = os.path.join(
        app.config['OUTPUT_DIR'], 'yolov8', 'annotatedMask.png')
    os.makedirs(os.path.dirname(preCleanedJSON), exist_ok=True)

    process_image(uploadPath, outputImage, preCleanedJSON,
                  maskOutput, confidence=0.25, overlap=0.5)

    #
    # This is for the JSON file correction (idk what to call it)
    #

    # output_data = transform_predictions(input_data, px_to_cm=6.923)
    
    try:
        with open(preCleanedJSON, 'r') as json_file:
            preCleanedJSON = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file '{preCleanedJSON}' was not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{preCleanedJSON}'.")
        exit(1)
    output_data = transform_predictions(preCleanedJSON)

    # Define the output file path using Flask's config
    CleanedJSON = os.path.join(
        app.config['OUTPUT_DIR'], 'yolov8', 'cleanedJSON', 'FloorplanPrediction.json')
    os.makedirs(os.path.dirname(CleanedJSON), exist_ok=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(CleanedJSON), exist_ok=True)

    # Save the transformed output to a JSON file
    with open(CleanedJSON, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Transformed output saved as '{CleanedJSON}'")

    #
    #   This is for the floorplan JSON to foundation JSON
    #
    import time 
    start = time.time()
    
    rl_module()
    
    end = time.time()
    
    lengthRLTimer = end - start
    print(f"Time took for {lengthRLTimer}")
    

    #
    # This is for sending the Features such as soil properties etc to the JSON
    #

    RLjsonfile = os.path.join(
        app.config['OUTPUT_DIR'], 'RL', 'RLfoundation.json')

    transfer_canvas_dimensions(CleanedJSON, RLjsonfile)

    #
    #   This is for the VAE part
    #

    seismic_zone,wind_speed,soil_type = (get_location_constants(location).values())
    
    soil_type_value = soil_type_map.get(soil_type)
    building_type_value = building_type_map.get(building_type)
    num_storey_value = num_storey_map.get(num_storey)
    material_spec_value = material_spec_map.get(material_spec)
    barsize_value = Barsize_map.get(barSize)
    
    print(soil_type)
    print(soil_type_value)
    
    building_type_value = int(building_type_value)
    num_storey_value = int(num_storey_value)
    material_spec_value = int(material_spec_value)
    barsize_value = int(barsize_value)
    soil_type_value=int(soil_type_value)
    

    single_input = [soil_type_value, material_spec_value, num_storey_value]

    # Call the ANN model with the prepared input

    column_scale, footing_scale = test_model(single_input)
    print(column_scale)
    

    foundationplanjson = os.path.join(
        app.config['OUTPUT_DIR'], 'RL', 'RLFoundation.json')
    generateFoundationPlan(foundationplanjson, column_scale, footing_scale,num_storey_value,barsize_value,location,roofType,lengthRLTimer,uploaded_file.filename)

    # TODO: timers

    # Provide feedback and return the image path as JSON
    return jsonify({'image_path': url_for('static', filename=f'images/final_combined_image.png')})


@app.route('/drawio')
def drawio():
    return render_template('drawio.html')


@app.route('/konva')
def konva():
    return render_template('konva.html')

# Route to save diagram to JSON


@app.route('/save', methods=['POST'])
def save():
    diagram_data = request.json.get('diagram')
    with open('diagram.json', 'w') as f:
        json.dump(diagram_data, f)
    return jsonify({'status': 'success', 'message': 'Diagram saved!'})

# Route to load diagram from JSON


@app.route('/load', methods=['GET'])
def load():
    if os.path.exists('diagram.json'):
        with open('diagram.json', 'r') as f:
            diagram_data = json.load(f)
        return jsonify({'status': 'success', 'diagram': diagram_data})
    else:
        return jsonify({'status': 'error', 'message': 'No saved diagram found!'})
    
    
@app.route('/save-foundation-plan', methods=['POST'])
def save_foundation_plan():
    data = request.json
    file_name = data.get('fileName', 'foundation-plan.cct')  # Default to foundation-plan.cct
    file_type = data.get('fileType', 'cct')  # Default to CCT

    json_file_path = os.path.join('output', 'RL', 'RLFoundationPadded.json')  # The original JSON file
    save_path = os.path.join('output', 'RL', file_name)  # Save in the output folder

    try:
        # Read the JSON data and save it as a .cct file (or other formats as needed)
        with open(json_file_path, 'r') as json_file:
            json_data = json_file.read()
        
        # You can add logic to transform the JSON into a proper .cct format if necessary,
        # but here it's assumed that you are just renaming the file.
        with open(save_path, 'w') as cct_file:
            cct_file.write(json_data)
        
        return jsonify({"message": "File saved successfully", "fileName": file_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

   
@app.route('/save-as-foundation-plan', methods=['POST'])
def save_as_foundation_plan():
    data = request.json
    file_type = data.get('fileType', 'png')  # Get the selected file type
    file_name = data.get('fileName', 'foundation-plan')  # Get the desired file name

    if file_type == 'cct':  # Save the JSON file as .cct
        json_file_path = os.path.join('output', 'RL', 'RLFoundationPadded.json')  # Path to the original JSON
        save_path = os.path.join('output', 'RL', file_name)  # Save path for the .cct file

        try:
            # Read the JSON data and save it as a .cct file (you may want to format or transform it here)
            with open(json_file_path, 'r') as json_file:
                json_data = json_file.read()

            with open(save_path, 'w') as cct_file:
                cct_file.write(json_data)

            return jsonify({"message": "File saved successfully", "fileName": file_name})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    elif file_type == 'png':  # Save the image as .png
        image_file_path = os.path.join('static', 'images', 'final_combined_image.png')  # Path to the PNG image
        save_path = os.path.join('output', 'RL', file_name)  # Save path for the .png file

        try:
            # Open and save the image as PNG
            img = Image.open(image_file_path)
            img.save(save_path, 'PNG')  # Save the image as PNG format

            return jsonify({"message": "Image saved successfully", "fileName": file_name})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    else:
        return jsonify({"error": "Unsupported file type."}), 400

@app.route('/download/<filename>')
def download_file(filename):
    # Define the directory where files are saved
    directory = os.path.join('output', 'RL')

    try:
        # Send the file to the user
        return send_from_directory(directory, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found."}), 404


if __name__ == '__main__':
    app.run(debug=True)
