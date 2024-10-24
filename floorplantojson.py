import os
import cv2
import json
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv

# TODO: make the flow clearer -> roboflow -> correctFormat
def process_image(image_path, output_image_path, json_output_path, mask_output_path, confidence=0.01, overlap=0.5):
    """
    Authenticate Roboflow, load image, run inference, visualize results, 
    save the annotated image, generate a black-and-white mask, and save JSON.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve environment variables
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    workspace = os.environ.get('WORKSPACE')
    project_name = os.environ.get('PROJECT')
    version = os.environ.get('VERSION', '6')  # Default to '6' if not set

    # Check for missing variables
    if not api_key or not workspace or not project_name:
        raise ValueError("Missing environment variables. Please set ROBOFLOW_API_KEY, WORKSPACE, and PROJECT.")

    # Authenticate and load the model
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    model = project.version(version).model

    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.imread(image_path)

    # Create a blank black-and-white mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Run inference
    results = model.predict(image_path, confidence=confidence, overlap=overlap).json()

    # Save results as JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Inference results saved as JSON at {json_output_path}")

    # Visualize results and update the mask
    for prediction in results.get('predictions', []):
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        label = prediction['class']

        # Calculate bounding box coordinates
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))

        # Draw the bounding box on the image
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the black-and-white mask
        cv2.rectangle(mask, start_point, end_point, 255, -1)  # White for walls

    # Save the annotated image
    cv2.imwrite(output_image_path, image)
    print(f"Annotated image saved at {output_image_path}")

    # Save the black-and-white mask
    cv2.imwrite(mask_output_path, mask)
    print(f"Black-and-white mask saved at {mask_output_path}")


# Example usage
if __name__ == "__main__":
    image_path = "testImages/cat1_2.jpg"  # Path to the input image
    output_image_path = "annotated_image.png"  # Path to save the annotated image
    json_output_path = "inference_results.json"  # Path to save the inference results as JSON
    mask_output_path = "mask_image.png"  # Path to save the black-and-white mask

    process_image(image_path, output_image_path, json_output_path, mask_output_path, confidence=0.25, overlap=0.5)