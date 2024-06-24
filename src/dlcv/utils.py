import os
import csv
import json
import random
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import yaml
from yacs.config import CfgNode as CN
from dlcv.config import get_cfg_defaults
from dlcv.dataset import *

# ToDo import functions

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def cfg_node_to_dict(cfg_node):
    """Convert a yacs CfgNode to a dictionary."""
    if not isinstance(cfg_node, CN):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v)
        return cfg_dict

def create_config(run_name, backbone, base_lr, batch_size, num_epochs,
                   horizontal_flip_prob, rotation_degrees, milestones, gamma,
                    pretrained_weights= '',
                    root='/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR',
                    config_dir='/kaggle/working/create_config'):
    # Get default configuration
    cfg = get_cfg_defaults()

    # Update the configuration with provided arguments
    cfg.DATA.ROOT = root
    cfg.MISC.RUN_NAME = run_name
    cfg.MODEL.BACKBONE = backbone
    cfg.TRAIN.BASE_LR = base_lr
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB = horizontal_flip_prob
    cfg.AUGMENTATION.ROTATION_DEGREES = rotation_degrees
    cfg.TRAIN.MILESTONES = milestones
    cfg.TRAIN.GAMMA = gamma
    cfg.MISC.PRETRAINED_WEIGHTS = pretrained_weights
    # Ensure the config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Define the path for the new config file
    config_file_path = os.path.join(config_dir, f'{run_name}.yaml')
    #os.makedirs(config_file_path, exist_ok=True)

     # Convert the config object to a dictionary
    cfg_dict = cfg_node_to_dict(cfg)
    
    # Save the updated configuration to a YAML file
    with open(config_file_path, 'w') as config_file:
        yaml.dump(cfg_dict, config_file, default_flow_style=False)

    print(f"Config file saved at: {config_file_path}")
    with open(config_file_path, 'r') as file:
        print(file.read())
    return config_file_path

def get_model(num_classes, backbone_name='resnet50', pretrained=True):
    # Define backbones and weights
    if backbone_name == 'resnet50':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone = torchvision.models.resnet50(weights=weights if pretrained else None)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        
    elif backbone_name == 'resnet101':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Update this line when FasterRCNN_ResNet101 weights are available
        backbone = torchvision.models.resnet101(weights=weights if pretrained else None)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        
    elif backbone_name == 'mobilenet':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(weights=weights if pretrained else None).features
        backbone.out_channels = 960
        
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")
    
    # Create the model using the specified backbone
    model = FasterRCNN(backbone, num_classes=num_classes)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

def load_pretrained_weights(network, weights_path, device):
    # Load the state_dict from the saved file
    state_dict = torch.load(weights_path, map_location=device)
    
    # Load the state_dict into the network
    network.load_state_dict(state_dict)
    
    return network
    pass # ToDo


def freeze_layers(network, frozen_layers):
    for name, param in network.named_parameters():
        if any(layer_name in name for layer_name in frozen_layers):
            param.requires_grad = False

def save_model(model, path):
    # Ensure device independence
    model_state = model.to(torch.device('cpu')).state_dict()
    torch.save(model_state, path + ".pth")

def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    param_groups = []
    for name, param in network.named_parameters():
        for layer_name, lr in stratification_rates.items():
            if name.startswith(layer_name):
                param_groups.append({'params': param, 'lr': lr})
                break
        else:
            param_groups.append({'params': param, 'lr': base_lr})
    return param_groups

def get_transforms(train):
    if train:
        return transforms.Compose([transforms.ToTensor(), RandomHorizontalFlip(p=0.5)])
    else:
        return transforms.Compose([transforms.ToTensor()])

def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])


def plot_metrics(metrics_file_path):
    # Load the metrics from the file
    if not os.path.exists(metrics_file_path):
        print(f"Metrics file not found: {metrics_file_path}")
        return
    
    with open(metrics_file_path, 'r') as f:
        metrics_history = json.load(f)

    epochs = range(1, len(metrics_history["train_loss"]) + 1)

    images_dir = "/kaggle/working/repository_content/images"

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics_history["train_loss"], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'training_loss.png'))
    plt.close()

    # Plot mAP
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics_history["mAP"], label='mAP')
    plt.plot(epochs, metrics_history["mAP_50"], label='mAP@50')
    plt.plot(epochs, metrics_history["mAP_75"], label='mAP@75')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'mAP.png'))
    plt.close()

    # Plot mAP for different sizes
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics_history["mAP_small"], label='mAP_small')
    plt.plot(epochs, metrics_history["mAP_medium"], label='mAP_medium')
    plt.plot(epochs, metrics_history["mAP_large"], label='mAP_large')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP for Different Object Sizes over Epochs')
    plt.legend()
    plt.savefig(os.path.join(images_dir, 'mAP_sizes.png'))
    plt.close()
    print(f"Plots saved in {images_dir}")


def visualize_inference_results(model, dataset, coco, device, num_images=5, output_dir='/kaggle/working/repository_content/images'):
    model.eval()
    # os.makedirs(output_dir, exist_ok=True)

    transform = transforms.ToTensor()
    dataset_size = len(dataset)
    random_indices = random.sample(range(dataset_size), num_images)

    for idx in random_indices:
        img, _ = dataset[idx]
        img_id = coco.getImgIds(imgIds=coco.dataset['images'][idx]['id'])[0]
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataset.root, 'TD-TSR/images/val', img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Convert image tensor to PIL image for visualization
        img = transforms.ToPILImage()(img_tensor.squeeze().cpu())

        # Plot image and bounding boxes
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score >= 0.5:  # Filter out low confidence detections
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                width, height = xmax - xmin, ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{label.item()} {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

        plt.title(f"Inference Result for Image ID {img_id}")
        save_path = os.path.join(output_dir, f'inference_{img_id}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Inference result saved for image {img_id} at {save_path}")


def generate_coco_results(model, test_image_dir, output_json_path, device):
    model = model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    results = []
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(test_image_dir):
        if image_name.endswith('.png'):
            img_path = os.path.join(test_image_dir, image_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img_tensor)[0]

            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score >= 0.5:  # Use a threshold to filter low confidence detections
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    width, height = xmax - xmin, ymax - ymin
                    results.append({
                        "file_name": image_name,
                        "category_id": int(label.item()),
                        "bbox": [float(xmin), float(ymin), float(width), float(height)],
                        "score": float(score.item())
                    })

    with open(output_json_path, 'w') as f:
        json.dump(results, f)

    print(f"COCO results saved to {output_json_path}")
