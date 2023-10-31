# Created: 21 July 23

import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import time
import argparse


# minimum size of bounding box after augmentation
min_box_size = 200

# Path to the .txt file containing filenames that need augmentation
txt_file = "/Users/ngoc/filenames/BA.txt"

def read_annotation_file(annotation_file_path, image_width, image_height):
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        label_id, x_center, y_center, width, height = map(float, line.strip().split())
        x_min = int((x_center - width / 2) * image_width)
        y_min = int((y_center - height / 2) * image_height)
        x_max = int((x_center + width / 2) * image_width)
        y_max = int((y_center + height / 2) * image_height)
        bboxes.append((label_id, x_min, y_min, x_max, y_max))

    return bboxes

def augment_and_save(image, bboxes, output_image_path, output_txt_path, augmentation, image_width, image_height, suffix, min_box_size):
    # Convert the bounding box coordinates to imgaug BoundingBox objects with labels
    bounding_boxes = [ia.BoundingBox(x1=bbox[1], y1=bbox[2], x2=bbox[3], y2=bbox[4], label=str(int(bbox[0]))) for bbox in bboxes]

    # Create the imgaug BoundingBoxesOnImage object
    bbs = ia.BoundingBoxesOnImage(bounding_boxes, shape=image.shape)

    # Perform augmentation on both the image and bounding boxes
    augmented_image, augmented_bbs = augmentation(image=image, bounding_boxes=bbs)

    # Filter out small or out-of-frame bounding boxes
    augmented_bbs = augmented_bbs.remove_out_of_image().clip_out_of_image()

    # Compute the size of each bounding box after augmentation
    augmented_bboxes_sizes = [(bbox.x2 - bbox.x1) * augmented_image.shape[1] * (bbox.y2 - bbox.y1) * augmented_image.shape[0] for bbox in augmented_bbs.bounding_boxes]

    # Create a new BoundingBoxesOnImage object with only large enough bounding boxes
    augmented_bbs = ia.BoundingBoxesOnImage([bbox for bbox, size in zip(augmented_bbs.bounding_boxes, augmented_bboxes_sizes) if size >= min_box_size], shape=augmented_image.shape)

    # Save the augmented image only if there are remaining bounding boxes
    if augmented_bbs.bounding_boxes:
        output_image_with_suffix = output_image_path.replace(".jpg", f"_{suffix}.jpg")
        cv2.imwrite(output_image_with_suffix, augmented_image)

        # Save the augmented bounding box coordinates back to the YOLO format annotation file
        output_txt_with_suffix = output_txt_path.replace(".txt", f"_{suffix}.txt")
        with open(output_txt_with_suffix, "w") as f:
            for bbox in augmented_bbs.bounding_boxes:
                x_center = (bbox.x1 + bbox.x2) / (2.0 * image_width)
                y_center = (bbox.y1 + bbox.y2) / (2.0 * image_height)
                width = (bbox.x2 - bbox.x1) / image_width
                height = (bbox.y2 - bbox.y1) / image_height
                f.write(f"{bbox.label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=int, required=True, help="Suffix for filenames")
    args = parser.parse_args()

    # Get the suffix from command-line arguments
    suffix = args.suffix

    # Set the path to the input image directory
    input_image_dir = "/Users/ngoc/train/images"

    # Set the path to the output (augmented) image directory
    output_image_dir = "/Users/ngoc/train_aug/images"
    
    # Set the path to the output (augmented) bounding box directory
    output_txt_dir = "/Users/ngoc/train_aug/labels"

    # Create the output directories if they don't exist
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)


    # Read the filenames from the .txt file
    with open(txt_file, "r") as f:
        filenames = [line.strip() for line in f]
    
    for i in range(len(filenames)):
        filenames[i] = filenames[i].replace(".txt", ".jpg")

    # Define augmentation sequence
    augmentation = iaa.Sequential([
        iaa.Rot90((1, 3)),
        iaa.Fliplr(0.5),  # Horizontal flip with a 50% probability
        iaa.Flipud(0.5),
        iaa.Sometimes(
            0.7,
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.Rotate((-20, 20)), # iaa.Affine(rotate=(-10, 10)), 
            #iaa.TranslateX(percent=(-0.1, 0.1)),
            #iaa.TranslateY(percent=(-0.1, 0.1)), #(px=(-20,20))
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1.0, 1.2)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
        ),
        iaa.Sometimes(
            0.4,
            iaa.Multiply((1, 1.2)),  # Multiply pixel values to adjust brightness
            iaa.GammaContrast((0.8, 1.5)),  # Apply gamma contrast adjustment
            iaa.GaussianBlur(sigma=(0.0, 1.5)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)),# Add Gaussian noise with mean=0 and std=0 to 5% of the pixel value range
        ),
        random_order=True  
        # You can adjust or add more augmentations here as per your requirement
    ])

    # Loop through each image in the input directory
    start_time = time.time()
    for filename in filenames:
        image_path = os.path.join(input_image_dir, filename)
        annotation_path = image_path.replace(".jpg", ".txt")  # Assuming annotations are in the same directory and have the same filename
        
        # Read the image
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Read the bounding box coordinates from the annotation file
        bboxes = read_annotation_file(annotation_path, image_width, image_height)

        # Augment the image and bounding boxes
        output_image_path = os.path.join(output_image_dir, filename)
        output_txt_path = os.path.join(output_txt_dir, filename.replace(".jpg", ".txt"))
        augment_and_save(image, bboxes, output_image_path, output_txt_path, augmentation, image_width, image_height, suffix, min_box_size)

    end_time = time.time()
    print("Augmentation and saving complete.")
    print(f"Execution time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()