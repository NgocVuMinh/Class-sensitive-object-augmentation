# Class-sensitive-object-augmentation
A simple class-sensitive augmentation for object detection.

The scripts in this repo are used to augment images that do not contain a specific class of object. In this case, they are blood cell images annotated with bounding boxes in YOLO format. Only images that do not contain platelets are augmented.
The images were taken from [Acevedo et al. 2019](https://data.mendeley.com/datasets/snkd93bnjr/1)
Bounding box annotations were created using [Roboflow](https://roboflow.com/)

**Requirements**
python==3.9
opencv-python==4.8
imgaug==0.4

**Scripts**
Run the following:
```
python count_instances.py # count the number of instances of each class from the annotation files
python no_object.py # look through all annotation files (YOLO format) and list out files that do not contain the chosen object
python run_augment.py BA # augmentation
python visualize_results.py # visualize the augmented images with bounding boxes
```

