# Created 21 July 23

"""
This script looks through all annotation files (YOLO format)
and create .txt file including files that do not contain a particular object
"""

import os

def has_object_of_class(annotation_file_path, class_to_check):
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        label_id = int(line.strip().split()[0])
        if label_id == class_to_check:
            return True

    return False

def get_initial_sign(filename):
    return filename.split("_")[0]

def main():
    # Set the path to the directory containing annotation files
    annotation_dir = "/Users/ngoc/test/augment_to_git/train/"

    # Set the class for which you want to find files without objects
    class_to_check = 7  # class ID that you want to exclude

    # Initialize a dictionary to store filenames without objects of the specified class grouped by initial signs
    filenames_by_initial_sign = {}

    # Loop through each annotation file in the directory
    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            annotation_path = os.path.join(annotation_dir, filename)
            if not has_object_of_class(annotation_path, class_to_check):
                initial_sign = get_initial_sign(filename)
                if initial_sign not in filenames_by_initial_sign:
                    filenames_by_initial_sign[initial_sign] = []
                filenames_by_initial_sign[initial_sign].append(filename)

    # Save the grouped filenames to a .txt file for each initial sign
    for initial_sign, filenames in filenames_by_initial_sign.items():
        output_file = f"{initial_sign}_no_{class_to_check}.txt" 
        filenames.sort()  # Sort filenames for each initial sign
        with open(output_file, "w") as f:
            for filename in filenames:
                f.write(f"{filename}\n")

        # Print the count of filenames without objects of the specified class for each initial sign
        total_files = len(os.listdir(annotation_dir))
        files_without_class_count = len(filenames)
        print(f"Files without class {class_to_check} for '{initial_sign}': {files_without_class_count} / {total_files} instances")

if __name__ == "__main__":
    main()