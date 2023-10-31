# Create 20 July 23

import os
import cv2

def visualize_and_save_images(input_image_dir, output_image_dir, label_map):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # Loop through each image in the input directory
    for filename in os.listdir(input_image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_image_dir, filename)
            annotation_path = os.path.join(input_label_dir, filename.replace(".jpg", ".txt"))
 
            # Visualize and save the image with bounding boxes
            visualize_and_save_single_image(image_path, annotation_path, output_image_dir, label_map)

def visualize_and_save_single_image(image_path, annotation_path, output_dir, label_map):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        label_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)

        label = label_map.get(label_id, f'Label {label_id}')

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Save the image with bounding boxes to the output directory
    output_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_image_path, image)

# Example usage:
input_image_dir = "/Users/ngoc/train_aug/images"  # Directory containing augmented images
input_label_dir = "/Users/ngoc/train_aug/labels"  # Directory containing augmented annotations
output_image_dir = "/Users/ngoc/visualized/"  # Directory to save the visualized images
label_map = {0: 'ba', 1: 'eo', 2: 'erb', 3: 'ig', 4: 'lym', 5: 'mono', 6: 'neut', 7: 'platelet'}

visualize_and_save_images(input_image_dir, output_image_dir, label_map)
