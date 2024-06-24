import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os
from yacs.config import CfgNode as CN
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This package internal functions should be used here
from dlcv.config import get_cfg_defaults, get_cfg_from_file
from dlcv.dataset import *
from dlcv.utils import *
from dlcv.training import train_and_evaluate_model

def sum(a, b):
    x=a+b
    return x

def main(cfg):

    if not os.path.exists(cfg.DATA.ROOT):
        print(f"Dataset root path does not exist: {cfg.DATA.ROOT}")
    else:
        print(f"Dataset root path exists: {cfg.DATA.ROOT}")

    config_file_path = cfg.CONFIG_FILE_PATH
    print(f"Using configuration file: {config_file_path}")
    print("Configuration for this run:")
    print(cfg.dump())

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu")
    print(f'Device: {device}')

    # Define transformations for training and testing

    train_transform = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB)
    ])
    test_transform = Compose([ToTensor()])
    

    # Load datasets
    train_dataset = CISOLDataset(root=cfg.DATA.ROOT, transforms=train_transform, train= True)
    test_dataset = CISOLDataset(root=cfg.DATA.ROOT, transforms=test_transform, train= False)

    # Initialize training and test loader
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = get_model(num_classes=cfg.MODEL.NUM_CLASSES)
    model.to(device)

    # Load pretrained weights if specified
    if cfg.MISC.PRETRAINED_WEIGHTS:
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)

    # Freeze layers if set as argument
    if cfg.MISC.FROZEN_LAYERS:
        freeze_layers(model, cfg.MISC.FROZEN_LAYERS)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.TRAIN.BASE_LR)

    # Define the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Define a scheduler - use the MultiStepLR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)

    coco_annotation_file = os.path.join(cfg.DATA.ROOT, 'TD-TSR/annotations/val.json')
    coco_gt = COCO(coco_annotation_file)


    # Hand everything to the train and evaluate model function
    train_losses, val_metrics, metrics_file_path = train_and_evaluate_model(model, train_loader, test_loader,
                                optimizer, cfg.TRAIN.NUM_EPOCHS,
                                device, coco_gt, scheduler=scheduler,
                                early_stopping=cfg.TRAIN.EARLY_STOPPING)

    # Plot metrics
    plot_metrics(metrics_file_path)

    visualize_inference_results(model, test_dataset, coco_gt, device)
    
    # Save the model using the default folder
    if cfg.MISC.SAVE_MODEL_PATH:
        save_model(model, cfg.MISC.SAVE_MODEL_PATH + "/" + cfg.MISC.RUN_NAME)
    
    config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + '_runConfig.yaml')
    with open(config_save_path, 'w') as f:
        f.write(cfg.dump())
    
    test_image_dir = os.path.join(cfg.DATA.ROOT, 'TD-TSR/images/test')
    output_json_path = os.path.join('/kaggle/working/output_json', cfg.MISC.RUN_NAME + '_coco_results.json')
    generate_coco_results(model, test_image_dir, output_json_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--config', type=str, help="Path to the config file")
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()

    # Check if the script is run with the command-line argument or from the notebook
    if args.config:
        config_file_path = args.config
    else:
        # Fallback to an environment variable or a default config path
        config_file_path = os.getenv('CONFIG_FILE', 'configs/default_config.yaml')

    # cfg.CONFIG_FILE_PATH = config_file_path
    # cfg.merge_from_file(config_file_path)
    cfg = get_cfg_from_file(config_file_path)
    main(cfg)
