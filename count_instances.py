# Created: 20 July 2023

"""
This script counts the number of instances of each class from the annotation files.
"""

import os

def count_instances(annotation_file_path):
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    with open(annotation_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        label_id = int(line.strip().split()[0])
        if label_id in class_counts:
            class_counts[label_id] += 1

    return class_counts

def main():
    # Set the path to the directory containing annotation files in YOLO format
    annotation_dir = "/Users/ngoc/test/images"

    # Initialize a dictionary to store the total count of instances for each class (7 classes in total)
    total_class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

    # Loop through each annotation file in the directory
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            annotation_path = os.path.join(annotation_dir, filename)
            class_counts = count_instances(annotation_path)

            # Update the total class counts
            for label_id, count in class_counts.items():
                total_class_counts[label_id] += count

    # Print the total class counts across all annotation files
    print("Total class counts:")
    for label_id, count in total_class_counts.items():
        print(f"Class {label_id}: {count} instances")

if __name__ == "__main__":
    main()
