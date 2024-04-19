# Total_Care
# Yolo. v5 Object detection project
# Participants: Seungwon Lee, Heejung Lim, Hyungwoo Lee


<img width="563" alt="Screenshot 2024-03-20 at 12 16 35 PM" src="https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5/assets/97718938/7ba2f834-3694-454a-aafc-2c79ffb1b141">

![YOLO confidence](https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5/assets/97718938/caff33ef-7d53-431c-9760-77c2df696ddf)
![YOLO lossval](https://github.com/Heisnotanimposter/Totalcare_autonomous_driving_Yolov5/assets/97718938/412cf962-48dd-44af-9e16-18e259b4ead9)

This repository contains code and instructions to train a YOLOv5 object detection model for [task description - e.g., traffic object detection, weather pattern identification].

## Requirements

* Python >= 3.6
* `torch` >= 1.7.0
* `ultralytics`
* Other dependencies in `requirements.txt`


## Installation

Contents
About
Requirements
Setup
Inference
Results
Known Issues
Additional Notes
About

This repository includes instructions, code, and pre-trained weights to implement a YOLOv5 object detection model for weather prediction.

Main Features

Can train on custom weather imagery
Offers flexibility in choosing YOLOv5 versions
Includes an inference example to test on new images
Requirements
Python >= 3.6
torch >= 1.7.0 (pip install torch torchvision)
ultralytics (pip install ultralytics)
Other dependencies listed in requirements.txt (pip install -r requirements.txt)
Setup
Clone repository:

Bash
git clone https://github.com/YourName/YOLOv5-weather-prediction
Use code with caution.
Download Data

Option 1: Public data: Source public weather datasets (e.g. search online, consider cloud types, weather conditions).
Option 2: Custom data: If needed, collect and annotate your weather images. Popular annotation tools include:
LabelImg: [https://github.com/tzutalin/labelImg]
CVAT: [https://github.com/openvinotoolkit/cvat]
Configure YOLOv5

Create a data.yaml file in the project directory, specifying paths to your training, validation, and test sets, label names, etc. Example:
YAML
train: ./path/to/train
val: ./path/to/val
test: ./path/to/test

nc: 3  # Number of classes (e.g., cloud types)
names: ['cirrus', 'cumulus', 'stratus'] 
Use code with caution.
(Optional) Pre-trained Weights:
If starting from pre-trained weights, download the .pt file and place it in the project directory.

Inference
To run inference on new images:

Bash
python detect.py --source /path/to/image_or_folder --weights /path/to/weights.pt
Use code with caution.
Replace /path/to/image_or_folder with the path to your image or folder containing images.
Replace /path/to/weights.pt with the path to your trained YOLOv5 model weights.
See detect.py for additional options and arguments.

Results
The model will output detected objects in your images, along with bounding boxes and class labels. The accuracy of the model will depend on the quality of your dataset and your YOLOv5 settings.

Example Output:

image.jpg:
  - cirrus (80% confidence) [x1, y1, x2, y2]
  - cumulus (92% confidence) [x1, y1, x2, y2] 
Known Issues
Data Quality: Model performance is highly dependent on the quality and diversity of your training data.
Overfitting: Ensure proper data splitting and regularization techniques to prevent overfitting to specific weather conditions in your training set.
Label Imbalance: If your dataset has certain cloud types represented more frequently than others, address this imbalance with techniques such as data augmentation or weighted loss.
Additional Notes
For best results, pre-process your images (resizing, normalization) before feeding them to the model.
Experiment with different YOLOv5 model architectures (e.g., YOLOv5s, YOLOv5m, YOLOv5l) for speed and performance trade-offs.
Integrate the predictions from the YOLOv5 model with your overall weather forecasting system.

1. Directory Creation and Moving Files

Bash
!mkdir /content/dataset/
!mv /content/* /content/dataset/
Use code with caution.
!mkdir /content/dataset/ This creates a directory called "dataset" within the "/content" directory. The ! at the beginning tells the notebook to run this line as a shell command.
!mv /content/* /content/dataset/: This command attempts to move all files from the root of the /content directory into the newly created /content/dataset directory. However, you'll likely get an error because you cannot move a directory into itself.
2. Importing and Image Path Handling

Python
from glob import glob   

img_path = glob('/content/dataset/export/images/*.jpg') 
img_path[0:5] 
print(len(img_path)) 
Use code with caution.
from glob import glob: Imports the glob function for file pattern matching.
img_path = glob('/content/dataset/export/images/*.jpg'): Creates a list of file paths of all JPEG images (*.jpg) within the specified directory.
img_path[0:5]: Displays the first five image paths in the list.
print(len(img_path)): Prints the total number of images found.
3. Splitting Data for Training and Validation

Python
from sklearn.model_selection import train_test_split

train_img_path, valid_img_path = train_test_split(img_path, test_size = 0.3)
print(len(train_img_path), len(valid_img_path))
Use code with caution.
from sklearn.model_selection import train_test_split: Imports the function to split data.
train_img_path, valid_img_path = train_test_split(...): Splits the img_path list into training (train_img_path) and validation (valid_img_path) sets with a 70/30 ratio (specified by test_size=0.3).
print(...): Prints the lengths of the newly created training and validation sets.
4. Creating List Files

Python
with open('/content/dataset/train.txt', 'w') as f:
  f.write('\n'.join(train_img_path)+'\n')

with open('/content/dataset/valid.txt', 'w') as f:
  f.write('\n'.join(valid_img_path)+'\n')
Use code with caution.
with open(...)...: These blocks of code create two text files:
train.txt: Contains the paths to all images in the training set (one path per line).
valid.txt: Contains the paths to all images in the validation set.
5. Preparing YOLOv5 Data Configuration (data.yaml)

Bash
!cat /content/dataset/data.yaml
Use code with caution.
!cat /content/dataset/data.yaml: This displays the contents of your 'data.yaml' file, which is essential for configuring YOLOv5 training.
Python
with open('/content/dataset/data.yaml', 'r') as f:
  data = yaml.safe_load(f)

data['train'] = '/content/dataset/train.txt' 
data['val'] = '/content/dataset/valid.txt' 

with open('/content/dataset/data.yaml', 'w') as f:
  yaml.dump(data, f)

!cat /content/dataset/data.yaml  
Use code with caution.
Reading YAML: The code reads the existing data.yaml file and loads it into a Python dictionary named data.
Modifying Paths: It updates the 'train' and 'val' keys of the dictionary with the paths to your newly created train.txt and valid.txt files.
Writing YAML: The modified dictionary is written back to the data.yaml file.
Verification: The updated data.yaml content is displayed.

contact
 - Projectleader : HyungwooLee 010-3917-0557 / - participant0: https://github.com/Heisnotanimposter / - participant1: https://github.com/heejjj
