# CISOL Dataset - Object Detection Deep Learning Model

## Introduction
This project aims to detect table elements n the CISOL Dataset Images, using the FasterRCNN Model with A ResNet50 Backbone


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
For this Project, FasterRCNN Model with a ResNet50 backbone was chosen.  
Using Faster R-CNN with ResNet50 for table element detection provides a balanced combination of accuracy and efficiency. ResNet50, with its 50-layer architecture, is deep enough to capture intricate features necessary for identifying detailed elements like headers, rows, columns, and cells, while remaining computationally efficient. This makes it suitable for tasks requiring high precision without demanding excessive computational resources, enabling faster training and inference times. This balance ensures robust performance, making it ideal for practical applications in structured environments like tables.  
### Pretrained Weights
For this project, the COCO.DEFAULT pretrained weights for the FasterRCNN_ResNet50 model were chosen, as they are a good start for this object detection task.  

## Configuration Parameters
#### The .yaml files contained in the `./configs` folder defines the hyper-parameters used to run the training Model on different Model Hyper-parameters:

The Configurations that are defined in the `config.py` file are changed by fetching the hyper-parameters from the configuration.yaml files and replacing the default configurations. When a Configuration file is not mentioned while running the `train.py` , the model is trained using the `default_config.yaml` file which is set as an Environment variable in the Kaggle Notebook.  
The configuration files change the following hyper-parameters during the training and evaluation cycles:  

#### - `config_first.yaml` : 
    **DATA**:
        DATASET: 'CISOL_TD-TSR'
        ROOT: '/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    **MODEL**:
        NUM_CLASSES: 6

    **MISC**:
        RUN_NAME: 'CISOL_TD-TSR_firstRun'
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

  
#### - `config_second.yaml` :
    **DATA**:
        DATASET: 'CISOL_TD-TSR'
        ROOT: '/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR'

    **MODEL**:
        NUM_CLASSES: 6

    **MISC**:
        RUN_NAME: 'CISOL_TD-TSR_firstRun'
        RESULTS_CSV: '/kaggle/working/repository_content/results'
        SAVE_MODEL_PATH: '/kaggle/working/repository_content/saved_models'
        PRETRAINED_WEIGHTS: '/kaggle/input/pretrained_coco_default/pytorch/coco_default_table_detection/1/CISOL_TD-TSR_firstRun.pth'

    **TRAIN**:
        BASE_LR: 0.0001
        MILESTONES: [10, 20]
        GAMMA: 0.1
        BATCH_SIZE: 4
        NUM_EPOCHS: 30
        EARLY_STOPPING: True

    **AUGMENTATION**:
        HORIZONTAL_FLIP_PROB: 0.5
        ROTATION_DEGREES: 11  

  
## Steps Involved in running the Kaggle Notebook:
### Step 1: Install the Github Repository:
The Github Repository is installed as a Python package using the !pip install git+{url} command where:  
url = f"https://{user}:{token}@github.com/BUW-CV/dlcv24-assignment-4-haaroonafroz"  
Using this method, the GitHub repository is installed along with all the dependencies mentioned in the `pyproject.toml` file.  
The Version of the Repository can be validated by confirming the commit ID while the package is being installed.  

### Step 2: Import the Repository Content:
The Repository content is imported using the 'os' and 'shutil' libraries by copying the content from the repository to the '/kaggle/working/' directory. This ensures that the repository content is easily accessible when called/referenced in the Notebook. The Notebook is also programmed to print the Repository content to ensure where the files are placed within the environment so that providing an absolute path is easy.  
![Repository Content](./src/dlcv/images/Repository_content.png)


### Step 3: Create a Config file:
Using the 'create_config' function from `utils.py`,  a configuration file (of type .yaml) is generated and placed in the working directory of the Notebook. This Configuration file can be used to run the Training and Evaluation function by setting it as the 'config_file_path' when running the `train.py` file.  
![Create Config](./src/dlcv/images/create_config.png)

### Step 4: Training and Evaluation:
The `train.py` main file is run by following these steps:
    - Set the path for the configuration file on which the training process is to be run. (config_file_path = config_first.yaml / config_second.yaml)
    - Set an environment variable (CONFIG_FILE) to ensure that in case the config_file_path is not read / provided, the training can be run on the default configuration file (default_config.yaml).
    - The `train.py` file receives the argument using the yacs package in the form of CfgNode. When a configuration file(.yaml) is passed in the Notebook, the configurations are appended and passed into the main function.
    - The 'subprocess' library is used to run the `train.py` file by passing the arguments.


## Kaggle Notebook:
### The Kaggle Notebook can be viewed using the following link:
[Kaggle Notebook](https://www.kaggle.com/haaroonafroz/dlcv-assignment4-haaroonafrozognawala)

# Results

## First Configuration:
#### Version: 6/9
The Model is run using the GPU P100 Accelerator for 40 Epochs.
### 1st Epoch Result:
Epoch [1/40], Train Loss: 1.9054, mAP: 0.0189  
{'mAP': 0.01885053552567899, 'mAP_50': 0.07702633022885952, 'mAP_75': 1.4349261013057828e-05, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.01885053552567899}  
Time: Training: 3Min 43sec, Evaluation: 57sec

### After 40 Epochs:
Epoch [40/40]: Train Loss: 0.3800, mAP: 0.0495  
{'mAP': 0.04951552858418789, 'mAP_50': 0.11867271553707044, 'mAP_75': 0.004091059491700044, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.04951552858418789}  
Time: Training: 3Min 32sec, Evaluation: 52sec  

#### Total Version Time: 13449.3 seconds

## Inference Images: First Run
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_20](./src/dlcv/images/inference_20_firstRun.png)
![inference_111](./src/dlcv/images/inference_111_firstRun.png)
![inference_157](./src/dlcv/images/inference_157_firstRun.png)
![inference_162](./src/dlcv/images/inference_162_firstRun.png)


## Training Loss over Epochs: First Run
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 40 epochs.  
![Training Loss](./src/dlcv/images/training_loss_firstRun.png)

## mAP Metrics Plotting: First Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_firstRun](./src/dlcv/images/mAP_firstRun.png)
![mAP_sizes_firstRun](./src/dlcv/images/mAP_sizes_firstRun.png)  

-------------------------------------------------------------------------------------------------------------
## Second Configuration: First Run
#### Version: 8/9
The Model is run using the GPU P100 Accelerator for 30 Epochs.  
The Trained Model weights from the First Run (6/9) were saved and used for the Second Run with little tweaks in the hyper-parameters to give a head start for this model.  
Early Stopping was enabled for the Second Run to avoid overfitting the training data, and it was interrupted after 4 Epochs.  
### 1st Epoch Result:
Epoch [1/30]: Train Loss: 0.4293, mAP: 0.0452  
{'mAP': 0.04518775256040265, 'mAP_50': 0.10924099098337131, 'mAP_75': 0.0029076957310496944, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.04518775256040265}  
Time: Training: 3Min 26sec, Evaluation: 51sec  

### After 4 Epochs:
Epoch [4/30]: Train Loss: 0.4060, mAP: 0.0590  
{'mAP': 0.05896068249905127, 'mAP_50': 0.13292791860778802, 'mAP_75': 0.005446731723532065, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.05896068249905127}  
Time: Training: 3Min 32sec, Evaluation: 52sec  

#### Total Version Time: 1720.5 seconds

## Inference Images: Second Run
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_40](./src/dlcv/images/inference_40_secondRun.png)
![inference_83](./src/dlcv/images/inference_83_secondRun.png)
![inference_124](./src/dlcv/images/inference_124_secondRun.png)
![inference_134](./src/dlcv/images/inference_134_secondRun.png)


## Training Loss over Epochs: Second Run
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 4 epochs.  
![Training Loss](./src/dlcv/images/training_loss_secondRun.png)

## mAP Metrics Plotting: Second Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_secondRun](./src/dlcv/images/mAP_secondRun.png)
![mAP_sizes_secondRun](./src/dlcv/images/mAP_sizes_secondRun.png)  

-------------------------------------------------------------------------------------------------------------
## Second Configuration: Second Run
#### Version: 9/9
The Model is run using the GPU P100 Accelerator for 30 Epochs.  
The Trained Model weights from the First Run (6/9) were saved and used for the Third Run with little tweaks in the hyper-parameters to give a head start for this model.  
Early Stopping was enabled for the Third Run to avoid overfitting the training data, and it was interrupted after 9 Epochs.  
### 1st Epoch Result:
Epoch [1/30]: Train Loss: 0.4286, mAP: 0.0568  
{'mAP': 0.05680025423540037, 'mAP_50': 0.14013209270793844, 'mAP_75': 0.003306624768904508, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.05680025423540037}  
Time: Training: 3Min 31sec, Evaluation: 53sec  

### After 9 Epochs:
Epoch [9/30]: Train Loss: 0.3752, mAP: 0.0577  
{'mAP': 0.057733940781760786, 'mAP_50': 0.1345127699625716, 'mAP_75': 0.005450655248061672, 'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.057733940781760786}  
Time: Training: 3Min 31sec, Evaluation: 53sec  

#### Total Version Time: 3365.9 seconds

## Inference Images: Third Run
The `visualize_inference_results` function in utils.py uses the saved model to generate the inference images from the trained model.

### Sample Predictions: (Table Elements)
![inference_65](./src/dlcv/images/inference_65_thirdRun.png)
![inference_77](./src/dlcv/images/inference_77_thirdRun.png)
![inference_82](./src/dlcv/images/inference_82_thirdRun.png)
![inference_89](./src/dlcv/images/inference_89_thirdRun.png)
![inference_105](./src/dlcv/images/inference_105_thirdRun.png)


## Training Loss over Epochs: Third Run
The `plot_metrics` function in utils.py uses the saved results to plot the Training losses over 9 epochs.  
![Training Loss](./src/dlcv/images/training_loss_thirdRun.png)

## mAP Metrics Plotting: Second Run
The `plot_metrics` function also uses the saved results to plot the mAP Metrics along the training process.  
![mAP_thirdRun](./src/dlcv/images/mAP_thirdRun.png)
![mAP_sizes_thirdRun](./src/dlcv/images/mAP_sizes_thirdRun.png)  
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
### Submission:
![evalai_submission](./src/dlcv/images/evalai_submission.png)
![evalai_leaderboard](./src/dlcv/images/evalai_leaderboard.png)