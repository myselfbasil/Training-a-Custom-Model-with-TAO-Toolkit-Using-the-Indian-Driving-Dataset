<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://github.com/myselfbasil/Training-a-Custom-Model-with-TAO-Toolkit-Using-the-Indian-Driving-Dataset./blob/923e8139af32cb030226413f2fd0aefa351f0783/assets/header_img.png"
      >
    </a>
  </p>
</div>

Here, I’m using Detectnet_v2 model architecture for training a custom model using TAO Toolkit

## 1. Initial Setups
Open up a terminal and follow the commands:

What I have did was to make a new directory called “project_dir” in “/home/basil”

```bash
cd /home/basil/
mkdir project_dir
cd project_dir
```

login to:

```bash
docker login nvcr.io
```

&

```bash
docker login
```

I have copied my dataset files to “/home/basil/Desktop/IDD_KITTI_FORMAT/” like the directory tree given below:

```bash
$IDD_KITTI_FORMAT
└── data
    ├── testing
		│		└── image_2
		│		└── label_1
    └── training
		    └── image_2
	  		└── label_1
```

Now download the notebook & necessary files using the wget link below and unzip it

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-getting-started/versions/5.3.0/zip -O tao-getting-started_5.3.0.zip
```

```bash
unzip tao-getting-started_5.3.0.zip
```

Run the notebook file in a Jupyter Notebook instance

```bash
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

To run the DetectNet_v2 example notebook, follow these steps:

1. **Open the TAO Launcher Starter Kit notebook in your browser** by clicking on the provided link.
2. **Navigate to the `notebooks` directory** in the left-hand file explorer.
3. **Open the `tao_launcher_starter_kit` directory**.
4. **Inside `tao_launcher_starter_kit`, locate and open the `detectnet_v2` directory**.
5. **Finally, open the `detectnet_v2.ipynb` notebook file** within the `detectnet_v2` directory.
6. **Once the notebook is open**, run the cells inside it sequentially by clicking on each cell and pressing `Shift + Enter` or using the `Run` button in the toolbar.

Note: Refer to the notebook file uploded below for references.

https://drive.google.com/file/d/1ujIrjgxboy0_HwoDW3RYDFWjGCXMhz6O/view?usp=sharing

## 2. Making the Required Changes:

As for my scenario, I had to uncomment this line of code in the first cell:

```bash
%env NOTEBOOK_ROOT=/home/basil/notebooks/tao_launcher_starter_kit/detectnet_v2
```

Then, I had to define my own local project directory:

```bash
os.environ["LOCAL_PROJECT_DIR"] = "/home/basil/Desktop/IDD_KITTI_FORMAT/"
```

For the 4th cell inside the notebook, I have made the necessary changes to the “.tao_mounts.json”

```bash
# Define the dictionary with the mapped drives
drive_map = {
    "Mounts": [
        # Mapping the data directory
        {
            "source": os.environ["LOCAL_PROJECT_DIR"],
            "destination": "/workspace/tao-experiments"
        },
        # Mapping the specs directory.
        {
            "source": os.environ["LOCAL_SPECS_DIR"],
            "destination": os.environ["SPECS_DIR"]
        },
        {
            "source": "/home",
            "destination": "/home"
        },
    ],
}
```

I have removed the docker ptions and added mounted “/home” directory inside the root of the docker file.

Run the 5th cell and you wil see the final “.tao_mounts.json” file.

Run the 6th and 7th cells. After that run the 12th cell to verify the dataset. I have modified it according to my need. I have attached it below for reference. Do the same for your dataset also.

## Pre-Processing the Dataset:

> I had faced a problem in which the testing dataset I had was having classes like "vehicle fallback" so I had to rename it properly back to "vehicle_fallback" same for "traffic_sign" class and "traffic_light"
> 

The below code is to correct it:

```python
# Code to correct the label files format error in the testing directory

import os
from tqdm import tqdm

def replace_text_in_files(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    
    for filename in tqdm(txt_files, desc="Processing files", unit="file", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)
        
        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if "vehicle fallback" in content or "traffic sign" in content or "traffic light" in content:
                updated_content = content.replace("vehicle fallback", "vehicle_fallback")
                updated_content = updated_content.replace("traffic sign", "traffic_sign")
                updated_content = updated_content.replace("traffic light", "traffic_light")

                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(updated_content)
            else:
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
        except Exception as e:
            pass

if __name__ == "__main__":
    input_directory_path = "/dataset/idd/IDD_KITTI_FORMAT/testing/label_1/"
    output_directory_path = "/home/basil/Desktop/IDD_KITTI_FORMAT/data/processed_labels/"
    replace_text_in_files(input_directory_path, output_directory_path)
```

Image augmentation & bounding box augmentation:

( You can use the same code for training set and testing set, just change the directories 😊 )

```python
# Code to augment the images and the bounding boxes for testing of Indian Driving Dataset (IDD)

from tqdm import tqdm
import cv2
import albumentations as A
import os

def read_label_file(label_path):
    bboxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            bbox = {
                'class': parts[0],
                'truncation': float(parts[1]),
                'occlusion': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                '3d_bbox': list(map(float, parts[8:15]))
            }
            bboxes.append(bbox)
    return bboxes

def write_label_file(label_path, bboxes):
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            line = f"{bbox['class']} {bbox['truncation']} {bbox['occlusion']} {bbox['alpha']} " \
                   f"{bbox['bbox'][0]} {bbox['bbox'][1]} {bbox['bbox'][2]} {bbox['bbox'][3]} " \
                   f"{' '.join(map(str, bbox['3d_bbox']))}\n"
            file.write(line)

def augment_image_and_bboxes(image_path, label_path, output_image_dir, output_label_dir, resize_height, resize_width):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    bboxes = read_label_file(label_path)
    
    albumentations_bboxes = []
    for bbox in bboxes:
        normalized_bbox = [
            min(max(bbox['bbox'][0] / original_width, 0), 1),
            min(max(bbox['bbox'][1] / original_height, 0), 1),
            min(max(bbox['bbox'][2] / original_width, 0), 1),
            min(max(bbox['bbox'][3] / original_height, 0), 1)
        ]
        
        albumentations_bboxes.append(normalized_bbox)
    
    transform = A.Compose([
        A.Resize(height=resize_height, width=resize_width)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    transformed = transform(image=image, bboxes=albumentations_bboxes, category_ids=[bbox['class'] for bbox in bboxes])
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_image_dir, f"{image_name}.jpg")
    cv2.imwrite(output_image_path, transformed['image'])
    
    for i, bbox in enumerate(bboxes):
        denormalized_bbox = [
            round(transformed['bboxes'][i][0] * original_width),
            round(transformed['bboxes'][i][1] * original_height),
            round(transformed['bboxes'][i][2] * original_width),
            round(transformed['bboxes'][i][3] * original_height)
        ]
        bbox['bbox'] = denormalized_bbox
    
    output_label_path = os.path.join(output_label_dir, f"{image_name}.txt")
    write_label_file(output_label_path, bboxes)

input_images_dir = '/dataset/idd/IDD_KITTI_FORMAT/testing/image_2/'
input_labels_dir = '/home/basil/Desktop/IDD_KITTI_FORMAT/data/processed_labels/'
output_images_dir = '/home/basil/Desktop/IDD_KITTI_FORMAT/data/testing/image_2/'
output_labels_dir = '/home/basil/Desktop/IDD_KITTI_FORMAT/data/testing/label_1/'

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

image_files = sorted(os.listdir(input_images_dir))

with tqdm(total=len(image_files), desc="Processing images") as pbar:
    for i, filename in enumerate(image_files):
        image_name = os.path.splitext(filename)[0]
        image_path = os.path.join(input_images_dir, filename)
        label_path = os.path.join(input_labels_dir, f"{image_name}.txt")
        augment_image_and_bboxes(image_path, label_path, output_images_dir, output_labels_dir, resize_height=340, resize_width=630)
        pbar.update(1)

print("Processing complete!")
```

---

As per the the resolution you have augmneted yout images into, you have to change the following part in your training file. For me its like:

```
augmentation_config {
  preprocessing {
    output_image_width: 640 # change this
    output_image_height: 320 # & this!
    min_bbox_width: 1.0
    min_bbox_height: 1.0
    output_image_channel: 3
  }
  spatial_augmentation {
    hflip_probability: 0.5
    zoom_min: 1.0
    zoom_max: 1.0
    translate_max_x: 8.0
    translate_max_y: 8.0
  }
  color_augmentation {
    hue_rotation_max: 25.0
    saturation_shift_max: 0.20000000298
    contrast_scale_max: 0.10000000149
    contrast_center: 0.5
  }
}
```

I have augmented my dataset to 630x340, but as for tao toolkit, the resolutions should be multiples of 16. So I slightly modified it to 640x320. A very slight modification is not a problem.

```bash
# verify
import os

DATA_DIR = os.environ.get('LOCAL_DATA_DIR')
num_training_images = len(os.listdir(os.path.join(DATA_DIR, "training/image_2")))
num_training_labels = len(os.listdir(os.path.join(DATA_DIR, "training/label_1")))
num_testing_images = len(os.listdir(os.path.join(DATA_DIR, "testing/image_2")))
print("Number of images in the train/val set. {}".format(num_training_images))
print("Number of labels in the train/val set. {}".format(num_training_labels))
print("Number of images in the test set. {}".format(num_testing_images))
```

You will get something similar like this as your output:

**`Number of images in the train/val set. 29833`** 

**`Number of labels in the train/val set. 29832`** 

**`Number of images in the test set. 7459`**

Go to: detectnet_v2_tfrecords_kitti_trainval.txt file and make the necessary changes to it as neede for your dataset: Mine for reference have been given below: and run the 13th cell too.

```
TFrecords conversion spec file for kitti training
kitti_config {
  root_directory_path: "/home/basil/Desktop/IDD_KITTI_FORMAT/data/training"
  image_dir_name: "image_2"
  label_dir_name: "label_1"
  image_extension: ".jpg"
  partition_mode: "random"
  num_partitions: 2
  val_split: 14
  num_shards: 10
}
image_directory_path: "/home/basil/Desktop/IDD_KITTI_FORMAT/data/training"
```

Run all the other cells till 18th one and I have made changes to the directory to which the pretrained model is being stored in the 19th cell. For reference, it has been given below:

```bash
# Download the pretrained model from NGC
!ngc registry model download-version nvidia/tao/pretrained_detectnet_v2:resnet18 \
    --dest /home/basil/Desktop/IDD_KITTI_FORMAT/pretrained_resnet18/
```

Go to detectnet_v2_train_resnet18_kitti.txt file and modify it according to your dataset and its path.

- Change your training file extension,
- Update the path to locate your downloaded pre-trained model (If you have changed it)

The detectnet_v2_train_resnet18_kitti.txt file have been uploaded for reference:

https://drive.google.com/file/d/1r6at_5sQCTqRelxB5OPna0dR5QgczIVY/view?usp=sharing

Note: I have changed it, If you are using my notebook, make sure to change the path.
If you have followed the documentation nicely, you are good to go for training. 

Run the training cells & wait for it to complete :) 

---

Done!

Here is my notion website: 

[https://basilshaji.notion.site/Training-a-Custom-Model-with-TAO-Toolkit-Using-the-Indian-Driving-Dataset-cc5adfe38f0e4f7bbae26f9b19635388?pvs=4](https://www.notion.so/Training-a-Custom-Model-with-TAO-Toolkit-Using-the-Indian-Driving-Dataset-cc5adfe38f0e4f7bbae26f9b19635388?pvs=21)

Use it for reference.

```
Matching predictions to ground truth, class 1/15.: 100%|█| 2368/2368 [00:00<00:0
Matching predictions to ground truth, class 2/15.: 100%|█| 1056/1056 [00:00<00:0
Matching predictions to ground truth, class 3/15.: 100%|█| 2372/2372 [00:00<00:0
Matching predictions to ground truth, class 4/15.: 100%|█| 2170/2170 [00:00<00:0
Matching predictions to ground truth, class 5/15.: 100%|█| 1196/1196 [00:00<00:0
Matching predictions to ground truth, class 6/15.: 100%|█| 411/411 [00:00<00:00,
Matching predictions to ground truth, class 7/15.: 100%|█| 71/71 [00:00<00:00, 3
Matching predictions to ground truth, class 8/15.: 100%|█| 44/44 [00:00<00:00, 1
Matching predictions to ground truth, class 9/15.: 100%|█| 38/38 [00:00<00:00, 1
Matching predictions to ground truth, class 10/15.: 100%|█| 1662/1662 [00:00<00:
Matching predictions to ground truth, class 12/15.: 100%|█| 23/23 [00:00<00:00, 
Matching predictions to ground truth, class 13/15.: 100%|█| 70/70 [00:00<00:00, 
Epoch 20/20
=========================

Validation cost: 0.000436
Mean average_precision (in %): 23.3940

+------------------+--------------------------+
|    class name    | average precision (in %) |
+------------------+--------------------------+
|      animal      |    1.4912280701754386    |
|   autorickshaw   |    44.982177963976085    |
|     bicycle      |    4.307592859629058     |
|       bus        |    39.40603878422854     |
|       car        |    43.78135941791249     |
|     caravan      |    32.703525360469996    |
|    motorcycle    |    66.59924999109562     |
|      person      |    37.32320131347138     |
|      rider       |     36.9367601416938     |
|  traffic_light   |           0.0            |
|   traffic_sign   |    5.329200713816099     |
|     trailer      |           0.0            |
|      train       |           0.0            |
|      truck       |    35.43120962697439     |
| vehicle_fallback |    2.618864737614625     |
+------------------+--------------------------+

Median Inference Time: 0.011044
2024-09-06 11:39:33,042 [TAO Toolkit] [INFO] root 2102: Evaluation metrics generated.
2024-09-06 11:39:33,061 [TAO Toolkit] [INFO] root 2102: Training loop completed.
2024-09-06 11:39:33,062 [TAO Toolkit] [INFO] root 2102: Saving trained model.
2024-09-06 11:39:33,678 [TAO Toolkit] [INFO] root 2102: Model saved.
```

---

**Made with 🫶🏻 by Basil**
Check out my medium guide here: [medium.com](https://medium.com/@basilshaji32/training-a-custom-model-with-tao-toolkit-using-the-indian-driving-dataset-f4ae538e5c45)

You can go through my notion website: [notion.com](https://basilshaji.notion.site/Training-a-Custom-Model-with-TAO-Toolkit-Using-the-Indian-Driving-Dataset-cc5adfe38f0e4f7bbae26f9b19635388?pvs=4)
