# CISOL Dataset - Object Detection Deep Learning Model

## Introduction
This project aims to detect table elements in the CISOL Dataset Images, using the FasterRCNN Model with variable backbones.


## Project Structure
- `pyproject.toml` : The toml file defines all the dependencies that need to be installed while installing the repository as a python package:
        - Dependencies: "pytest", "torch", "tqdm", "matplotlib", "torchvision", "scikit-learn", "seaborn", "yacs", "pyyaml", "pycocotools"
- `config.py` : Script that contains the default configurations for runnning the Main file training and Evaluation
- `train.py` : Script to train the model (Main).
- `training.py` : Contains definition of functions to Train and Evaluate the model.
- `utils.py`: Contains definitions of the utility functions used throughout the project. 
- `configs/`: Directory containing model configuration files. (.yaml files)
- `saved_models/`: Directory to save trained model weights.
- `results/`: Directory to save training results and logs.

## Model
For this Project, FasterRCNN Model with 3 different backbones was chosen.  
1. FasterRCNN with Resnet50 Backbone:  
Using Faster R-CNN with ResNet50 for table element detection provides a balanced combination of accuracy and efficiency. ResNet50, with its 50-layer architecture, is deep enough to capture intricate features necessary for identifying detailed elements like headers, rows, columns, and cells, while remaining computationally efficient. This makes it suitable for tasks requiring high precision without demanding excessive computational resources, enabling faster training and inference times. This balance ensures robust performance, making it ideal for practical applications in structured environments like tables.  
2. FasterRCNN with Resnet101 Backbone:  
Using the FasterRCNN object detection model with a ResNet101 backbone offers high accuracy and robustness due to its deep architecture and extensive feature extraction capabilities. This makes it suitable for complex and high-resolution images.  
3. FasterRCNN with MobileNet Backbone:  
Using a MobileNet backbone with FasterRCNN provides a lightweight and efficient model, ideal for real-time applications and deployment on devices with limited computational resources, while still maintaining reasonable accuracy.  
### Pretrained Weights
For this project, the following weights were used for each backbone configuration:  
  
|    Backbone    |    Default Weights                                  |
|:--------------:|:---------------------------------------------------:|
|   Resnet50     |  FasterRCNN_ResNet50_FPN_Weights.DEFAULT            |
|   Resnet101    |  FasterRCNN_ResNet50_FPN_Weights.DEFAULT            |
|   MobileNet    |  FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT  |



## Steps Involved in running the Kaggle Notebook:
### Step 1: Install the Github Repository:
The Github Repository is installed as a Python package using the !pip install git+{url} command where:  
url = f"https://{user}:{token}@github.com/BUW-CV/dlcv24-individual-final-project-haaroonafroz"  
Using this method, the GitHub repository is installed along with all the dependencies mentioned in the `pyproject.toml` file.  
The Version of the Repository can be validated by confirming the commit ID while the package is being installed.  

### Step 2: Import the Repository Content:
The Repository content is imported using the 'os' and 'shutil' libraries by copying the content from the repository to the '/kaggle/working/' directory. This ensures that the repository content is easily accessible when called/referenced in the Notebook. The Notebook is also programmed to print the Repository content to ensure where the files are placed within the environment so that providing an absolute path is easy.  
![Repository Content](./images/Repository_content.png)


### Step 3: Create a Config file:
Using the 'create_config' function from `utils.py`,  a configuration file (of type .yaml) is generated and placed in the working directory of the Notebook. This Configuration file can be used to run the Training and Evaluation function by setting it as the 'config_file_path' when running the `train.py` file.  
![Create Config](./images/create_config_utils1.png)
![Create Config](./images/create_config_utils2.png)
![Create Config](./images/create_config_kaggle1.png)
![Create Config](./images/create_config_kaggle1.png)

### Step 4: Training and Evaluation:
The `train.py` main file is run by following these steps:
    - Set the path for the configuration file on which the training process is to be run. (config_file_path = create_config())
    - Set an environment variable (CONFIG_FILE) to ensure that in case the config_file_path is not read / provided, the training can be run on the default configuration file (default_config.yaml).
    - The `train.py` file receives the argument using the yacs package in the form of CfgNode. When a configuration file(.yaml) is passed in the Notebook, the configurations are appended and passed into the main function.
    - The 'subprocess' library is used to run the `train.py` file by passing the arguments.


## Kaggle Notebook:
### The Kaggle Notebook can be viewed using the following link:
[Kaggle Notebook](https://www.kaggle.com/code/haaroonafroz/dlcv-individual-project-haaroonafrozognawala)


# Configuration Parameters  

The Configurations that are defined in the `config.py` file are changed by fetching the hyper-parameters from the configuration.yaml files and replacing the default configurations. When a Configuration file is not mentioned while running the `train.py` , the model is trained using the `default_config.yaml` file which is set as an Environment variable in the Kaggle Notebook.  
The configuration files change the following hyper-parameters during the training and evaluation cycles:  

#### - `resnet50_config.yaml` : 
    **DATA**:
        DATASET: 'CISOL_TD-TSR'
        ROOT: '/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    **MODEL**:
        BACKBONE: resnet50
        NUM_CLASSES: 6

    **MISC**:
        RUN_NAME: 'resnet50_config'
        RESULTS_CSV: '/kaggle/working/repository_content/results'
        SAVE_MODEL_PATH: '/kaggle/working/repository_content/saved_models'
        PRETRAINED_WEIGHTS: ''

    **TRAIN**:
        BASE_LR: 0.0005
        MILESTONES: [10, 20]
        GAMMA: 0.1
        BATCH_SIZE: 4
        NUM_EPOCHS: 40
        EARLY_STOPPING: False

    **AUGMENTATION**:
        HORIZONTAL_FLIP_PROB: 0.5
        ROTATION_DEGREES: 10

  
#### - `resnet101_config.yaml` :
    **DATA**:
        DATASET: 'CISOL_TD-TSR'
        ROOT: '/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    **MODEL**:
        NUM_CLASSES: 6
        BACKBONE: resnet101

    **MISC**:
        RUN_NAME: 'resnet101_config'
        RESULTS_CSV: '/kaggle/working/repository_content/results'
        SAVE_MODEL_PATH: '/kaggle/working/repository_content/saved_models'
        PRETRAINED_WEIGHTS: ''
        
    **TRAIN**:
        BASE_LR: 0.0005
        MILESTONES: [10, 20]
        GAMMA: 0.1
        BATCH_SIZE: 2
        NUM_EPOCHS: 40
        EARLY_STOPPING: False

    **AUGMENTATION**:
        HORIZONTAL_FLIP_PROB: 0.5
        ROTATION_DEGREES: 10  


#### - `mobilenet_config.yaml` :
    **DATA**:
        DATASET: 'CISOL_TD-TSR'
        ROOT: '/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    **MODEL**:
        NUM_CLASSES: 6
        BACKBONE: mobilenet

    **MISC**:
        RUN_NAME: 'mobilenet_config'
        RESULTS_CSV: '/kaggle/working/repository_content/results'
        SAVE_MODEL_PATH: '/kaggle/working/repository_content/saved_models'
        PRETRAINED_WEIGHTS: ''
        
    **TRAIN**:
        BASE_LR: 0.0005
        MILESTONES: [10, 20]
        GAMMA: 0.1
        BATCH_SIZE: 8
        NUM_EPOCHS: 60
        EARLY_STOPPING: False

    **AUGMENTATION**:
        HORIZONTAL_FLIP_PROB: 0.5
        ROTATION_DEGREES: 10  

# Results  
## FasterRCNN-ResNet50
#### Version: 6/7
The Model is run using the GPU P100 Accelerator for 70 Epochs.  
| Results   |   Epoch: 1/70   |   Epoch: 40/70    |   Epoch: 70/70 |
|:---------:|:---------------:|:-----------------:|:--------------:|
|Training Loss| 2.2911 | 0.5665| 0.3994 |
|mAP|  1.026036479594107e-06 | 0.06002088844494126 | 0.0658 |
|mAP_50|  5.514755089236659e-06 | 0.1502604897974755 | 0.1557400544118862 |
|mAP_75|  0.0 | 0.0022192964970503366 | 0.0022192964970503366|
|mAP_small| 0.0 | 0.0 | 0.0 |
|mAP_medium| 0.0 | 0.0 | 0.0 |
|mAP_large| 1.026036479594107e-06 | 0.06002088844494126 | 0.06578527159756851 |
|Training Time| 3Min 59sec | 3Min 46sec | 04Min 27sec | 
|Evaluation Time| 52sec | 50sec | 01Min 01sec |  

## Inference Images: First Run
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_129](./images/inference_129_Resnet50_firstRun.png)
![inference_112](./images/inference_112_Resnet50_firstRun.png)
![inference_43](./images/inference_43_Resnet50_firstRun.png)
![inference_47](./images/inference_47_Resnet50_firstRun.png)


## Training Loss over Epochs: First Run
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 40 epochs.  
![Training Loss](./images/training_loss_Resnet50_firstRun.png)

## mAP Metrics Plotting: First Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_firstRun](./images/mAP_Resnet50_firstRun.png)
![mAP_sizes_firstRun](./images/mAP_sizes_Resnet50_firstRun.png)  

-------------------------------------------------------------------------------------------------------------
## Second Configuration: ResNet101
#### Version: 3/7
The Model is run using the GPU P100 Accelerator for 40 Epochs.  

### 1st Epoch Result:
Epoch [1/40]: Train Loss: 1.7978, mAP: 0.0471  
{'mAP': 0.04710195442291161, 'mAP_50': 0.11176092292190208, 'mAP_75': 0.0191188322210266, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.04710195442291161}  
Time: Training: 4Min 55sec, Evaluation: 1Min 02sec  

### After 40 Epochs:
Epoch [40/40]: Train Loss: 0.4759, mAP: 0.0638  
{'mAP': 0.06384850166383267, 'mAP_50': 0.154462482014483, 'mAP_75': 0.002926961590245225, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.06384850166383267}  
Time: Training: 4Min 34sec, Evaluation: 57sec  

#### Total Version Time: 15775.1 seconds

## Inference Images: Second Configuration
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_119](./images/inference_119_Resnet101_secondRun.png)
![inference_105](./images/inference_105_Resnet101_secondRun.png)
![inference_48](./images/inference_48_Resnet101_secondRun.png)
![inference_59](./images/inference_59_Resnet101_secondRun.png)


## Training Loss over Epochs: Second Configuration
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 4 epochs.  
![Training Loss](./images/training_loss_Resnet101_secondRun.png)

## mAP Metrics Plotting: Second Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_secondRun](./images/mAP_Resnet101_secondRun.png)
![mAP_sizes_secondRun](./images/mAP_sizes_Resnet101_secondRun.png)  

-------------------------------------------------------------------------------------------------------------
## Third Configuration: MobileNet
#### Version: 4/7
The Model is run using the GPU P100 Accelerator for 60 Epochs.    
### 1st Epoch Result:
Epoch [1/60]: Train Loss: 1.9932, mAP: 0.0004  
{'mAP': 0.0003530948519792429, 'mAP_50': 0.0013985628938955675, 'mAP_75': 0.0, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.0003530948519792429}  
Time: Training: 3Min 02sec, Evaluation: 47sec  

### After 60 Epochs:
Epoch [60/60]: Train Loss:  0.5849, mAP: 0.0606  
{'mAP': 0.06384850166383267, 'mAP_50': 0.154462482014483, 'mAP_75': 0.002926961590245225, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.06384850166383267}  
Time: Training: 2Min 51sec, Evaluation: 45sec  

#### Total Version Time: 16193.8 seconds

## Inference Images: Third Configuration
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_43](./images/inference_43_Mobilenet_thirdRun.png)
![inference_27](./images/inference_27_Mobilenet_thirdRun.png)
![inference_101](./images/inference_101_Mobilenet_thirdRun.png)
![inference_130](./images/inference_130_Mobilenet_thirdRun.png)


## Training Loss over Epochs: Third Configuration
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 9 epochs.  
![Training Loss](./images/training_loss_Mobilenet_thirdRun.png)

## mAP Metrics Plotting: Second Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_thirdRun](./images/mAP_Mobilenet_thirdRun.png)
![mAP_sizes_thirdRun](./images/mAP_sizes_Mobilenet_thirdRun.png)  

---------------------------------------------------------------------------------------------------  
# Interactive Visualization: Using the Notebook as a Pytorch Package
After the first 3 training runs, the saved models for each respective training runs were used to run one more round of training and evaluation.  
For these runs, an interactive method for visualizing the inference results was implemented:  
![get_model](./images/get_model.png)  
![visualize_inference_results](./images/visualize_inference_results.png)  

# Conclusion
After training the model with 3 different backbone configurations, the generated JSON file was pushed to `Eval.ai` and the following results were observed on the test dataset:  

1. FasterRCNN Model (ResNet50 Backbone):  
`First Run`: Version 2/7- 40 Epochs  
"mAP": 51.321747138025046  
"mAP IoU=.50": 60.89690064127804  
"mAP IoU=.75": 56.43127286619124  
`Second Run`: Version 6/7- 30 Epochs  
"mAP": 53.030737024040775  
"mAP IoU=.50": 61.29460891530856  
"mAP IoU=.75": 58.19626560674476  

3. FasterRCNN Model (ResNet101 Backbone):  
`First Run`: Version 4/7- 40 Epochs  
"mAP": 51.16211602905256  
"mAP IoU=.50": 60.2160120654103  
"mAP IoU=.75": 56.034203458428856  
`Second Run`: Version 7/7- 30 Epochs  
"mAP": 51.16211602905256  
"mAP IoU=.50": 60.2160120654103  
"mAP IoU=.75": 56.034203458428856  

4. FaterRCNN Model (MobileNet Backbone):  
`First Run`: Version 4/7- 60 Epochs  
"mAP": 51.24333915895759  
"mAP IoU=.50": 61.62360245036874  
"mAP IoU=.75": 56.9741775965515  
`Second Run`: Version 5/7- 30 Epochs  
"mAP": 55.7283211853313  
"mAP IoU=.50": 65.0254162650605  
"mAP IoU=.75": 60.1306060659238  
  
  
From the above results, we can observe that the FasterRCNN Model performs the best Object Detection task for detecting Table Elements, when trained with the `ResNet101` backbone for 70 Epochs.  
--------------------------------------------------------------------------------------------------

# Eval.ai Submission
The `utils.py` file contains the `generate_coco_results` function which defines the method to use the trained model and make predictions on the 'Test' dataset. These predictions are saved in a JSON file in the working directory of the Kaggle Notebook.  
This JSON file is of the format:  
        "file_name": image_name,  
        "category_id": int(label.item()),  
        "bbox": [float(xmin), float(ymin), float(width), float(height)],  
        "score": float(score.item())  

which is the same as mentioned in the submission guidlines.  

#### Eval.ai username: haaroonafroz  
### Submissions:  
A total of 6 submissions were made for the Indivisual Final Prject, as can be seen in the following photo.  
![evalai_submission](./images/evalai_submissions.png)  
#### Leaderboard with the most Promisisng Result:  
![evalai_leaderboard](./images/evalai_leaderboard.png)
