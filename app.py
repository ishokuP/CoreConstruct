from flask import Flask, render_template, request, jsonify, flash, redirect,url_for
import os
import json
import numpy as np
from werkzeug.utils import secure_filename

# Importing coreconstruct features

from floorplantojson import generate_mask_measure_walls_annotate
from addfeature import update_json_features
from JSONReinforcement import mainReinforcement
from foundationJSONtoimage import generate_separate_images

# Define the mappings
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

app = Flask(__name__)
app.secret_key = 'wowCoreandConstruct'

# Folder where uploaded files will be stored
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
OUTPUT_DIR='output/'


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
    soil_type = request.form.get('soilType')
    building_type = request.form.get('buildingType')
    num_storey = request.form.get('numStorey')
    material_spec = request.form.get('materialSpecs')

    # Print the values received for debugging
    print(f"Uploaded File: {uploaded_file.filename if uploaded_file else 'No file uploaded'}")
    print(f"Soil Type: {soil_type}")
    print(f"Building Type: {building_type}")
    print(f"Number of Storeys: {num_storey}")
    print(f"Material Spec: {material_spec}")

    # Perform backend validation
    missing_fields = []

    if not uploaded_file:
        print("Missing: Floor Plan")  # Debugging message
        missing_fields.append("Floor Plan")

    if not soil_type:
        print("Missing: Soil Type")  # Debugging message
        missing_fields.append("Soil Type")

    if not building_type:
        print("Missing: Building Type")  # Debugging message
        missing_fields.append("Building Type")

    if not num_storey:
        print("Missing: Number of Storeys")  # Debugging message
        missing_fields.append("Number of Storeys")

    if not material_spec:
        print("Missing: Material Specifications")  # Debugging message
        missing_fields.append("Material Specifications")

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
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the uploaded file with the original name
    uploaded_file.save(file_path)
    
    # 
    # This is for the Floorplan to JSON Part
    # 
    json_output_path = os.path.join(app.config['OUTPUT_DIR'], 'yolov8', 'output.json')
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    
    generate_mask_measure_walls_annotate(file_path, json_output_path)
    
    # 
    # This is for sending the Features such as soil properties etc to the JSON 
    # 
    update_json_features(json_output_path, soil_type, building_type, num_storey, material_spec)
    
    
    #     
    #   This is for the floorplan JSON to foundation JSON  
    #     

    mainReinforcement()
    
    
    # 
    #   This is for the VAE part 
    # 
    
    foundationplanjson=os.path.join(app.config['OUTPUT_DIR'], 'RL', 'RLFoundation.json')
    from ANN import test_model
    soil_type_value = soil_type_map.get(soil_type)
    building_type_value = building_type_map.get(building_type)
    num_storey_value = num_storey_map.get(num_storey)
    material_spec_value = material_spec_map.get(material_spec)
    
    soil_type_value = int(soil_type_value)
    building_type_value = int(building_type_value)
    num_storey_value = int(num_storey_value)
    material_spec_value = int(material_spec_value)
    
    single_input = [soil_type_value, material_spec_value, num_storey_value] 
    # Call the ANN model with the prepared input
    scale_1,scale_2 = test_model(single_input)
    footingexpandinflation = 75*scale_2
    generate_separate_images(foundationplanjson, footingexpandinflation, 35)
    
    
    foundationplanjson=os.path.join(app.config['OUTPUT_DIR'], 'RL', 'RLFoundation.json')
    final_output_path = 'static/images/final_combined.png'  # Save to static/images/
    # Provide feedback and return the image path as JSON
    return jsonify({'image_path': url_for('static', filename=f'images/final_combined.png')})



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

if __name__ == '__main__':
    app.run(debug=True)
