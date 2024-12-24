import os
import cv2

def verify_yolo_annotations(image_folder, annotation_folder, output_folder):
    """
    Verifies YOLO annotation files against a folder of images and saves annotated images.

    Parameters:
        image_folder (str): Path to the folder containing images.
        annotation_folder (str): Path to the folder containing YOLO .txt files.
        output_folder (str): Path to the folder to save annotated images.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_base_names = {os.path.splitext(f)[0] for f in image_files}

    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".txt")]
    annotation_base_names = {os.path.splitext(f)[0] for f in annotation_files}

    # Check for mismatches
    missing_annotations = image_base_names - annotation_base_names
    missing_images = annotation_base_names - image_base_names

    if missing_annotations:
        print(f"Missing annotations for images: {missing_annotations}")
    if missing_images:
        print(f"Missing images for annotations: {missing_images}")

    # Verify bounding boxes and save annotated images
    for annotation_file in annotation_files:
        base_name = os.path.splitext(annotation_file)[0]
        annotation_path = os.path.join(annotation_folder, annotation_file)
        image_path = os.path.join(image_folder, f"{base_name}.jpg")

        # Skip if the corresponding image doesn't exist
        if not os.path.exists(image_path):
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        height, width, _ = image.shape

        # Read and verify the annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid format in {annotation_file} on line {line_number}: {line}")
                continue

            class_id, center_x, center_y, box_width, box_height = map(float, parts)

            # Check bounding box validity
            if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                print(f"Invalid center coordinates in {annotation_file} on line {line_number}: {line}")
            if not (0 <= box_width <= 1 and 0 <= box_height <= 1):
                print(f"Invalid box dimensions in {annotation_file} on line {line_number}: {line}")

            # Calculate box coordinates
            x_min = int((center_x - box_width / 2) * width)
            y_min = int((center_y - box_height / 2) * height)
            x_max = int((center_x + box_width / 2) * width)
            y_max = int((center_y + box_height / 2) * height)

            # Draw bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"Class {int(class_id)}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save the annotated image
        output_path = os.path.join(output_folder, f"{base_name}_annotated.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")

    print("Verification and saving completed!")

# Example usage
image_folder = "./data/2nd/imgs"  # Path to the folder containing images
annotation_folder = "./data/2nd/lbl"  # Path to the folder containing YOLO .txt files
output_folder = "./data/2nd/annotated_check"  # Path to save annotated images

verify_yolo_annotations(image_folder, annotation_folder, output_folder)

