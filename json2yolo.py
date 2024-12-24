import json
import os

def convert_json_folder_to_yolo(input_folder, output_folder, class_mapping):
    """
    Converts all JSON annotation files in a folder to YOLO .txt format.

    Parameters:
        input_folder (str): Path to the folder containing JSON files.
        output_folder (str): Path to the folder to save YOLO .txt files.
        class_mapping (dict): Mapping of class labels to YOLO class IDs.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all JSON files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)
            
            # Process the JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get image dimensions
            image_width = data["imageWidth"]
            image_height = data["imageHeight"]
            
            # Prepare output YOLO .txt file path
            image_base_name = os.path.splitext(data["imagePath"])[0]
            output_file = os.path.join(output_folder, f"{image_base_name}.txt")
            
            # Process each shape in the JSON file
            yolo_lines = []
            for shape in data["shapes"]:
                label = shape["label"]
                points = shape["points"]
                
                # Convert label to class ID
                class_id = class_mapping.get(label, -1)
                if class_id == -1:
                    print(f"Skipping unknown label: {label}")
                    continue
                
                # Calculate YOLO format values
                x_min, y_min = points[0]
                x_max, y_max = points[1]
                center_x = ((x_min + x_max) / 2) / image_width
                center_y = ((y_min + y_max) / 2) / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # Add to YOLO annotations
                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            
            # Write YOLO annotations to the output file
            with open(output_file, 'w') as f:
                f.write("\n".join(yolo_lines))
            
            print(f"Converted: {filename} -> {output_file}")

# Example usage
input_folder = "./data/2nd/json"  # Folder containing JSON files
output_folder = "data/2nd/lbl"  # Folder to save YOLO .txt files
class_mapping = {"Hand_Raised": 0}  # Map class labels to YOLO class IDs

# Convert all JSON files in the folder
convert_json_folder_to_yolo(input_folder, output_folder, class_mapping)
