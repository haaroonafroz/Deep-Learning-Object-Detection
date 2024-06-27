import torch
from tqdm import tqdm
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

def train_one_epoch(model, data_loader, optimizer, device):

    model.train()  # Set the model to training mode
    epoch_loss = 0.0

    # Iterate over the data loader
    for images, targets in tqdm(data_loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()  # Zero out gradients

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()  # Update model parameters
        
        epoch_loss += losses.item() * len(images)  # Accumulate loss

    epoch_loss /= len(data_loader.dataset)  # Calculate average loss
    return epoch_loss


def evaluate_one_epoch(model, data_loader, coco_gt, device):
    model.eval()
    epoch_loss = 0.0
    results = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass for predictions
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                if len(boxes) == 0:
                    print(f"No boxes detected for image_id {image_id}")
                    continue

                for j in range(boxes.shape[0]):
                    box = boxes[j]
                    score = scores[j]
                    label = labels[j]
                    results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": box.tolist(),
                        "score": score
                    })

    if not results:
        print("No results to evaluate!")
        return 0.0, {}

    #epoch_loss /= len(data_loader.dataset)

    # Save results to file and evaluate with COCO
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Collect metrics
    metrics = {
        "mAP": coco_eval.stats[0],  # Mean Average Precision
        "mAP_50": coco_eval.stats[1],  # Mean Average Precision at IoU=0.50
        "mAP_75": coco_eval.stats[2],  # Mean Average Precision at IoU=0.75
        "mAP_small": coco_eval.stats[3],  # Mean Average Precision for small objects
        "mAP_medium": coco_eval.stats[4],  # Mean Average Precision for medium objects
        "mAP_large": coco_eval.stats[5],  # Mean Average Precision for large objects
    }
    return 0.0, metrics

def train_and_evaluate_model(model, train_loader, val_loader, optimizer, num_epochs, device, coco_gt, scheduler=None, early_stopping=False):
    train_losses = []
    #val_losses = []
    val_metrics = []
    best_val_map = -float('inf')
    consecutive_no_improvement = 0
    metrics_history = {"train_loss": [], "mAP": [], "mAP_50": [], "mAP_75": [], "mAP_small": [], "mAP_medium": [], "mAP_large": []}
    
    #Save the Metrics in /results
    results_dir = "/kaggle/working/results"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, "metrics_history.json")


    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        _, metrics = evaluate_one_epoch(model, val_loader, coco_gt, device)  # No loss returned during evaluation
        val_metrics.append(metrics)

         # Track metrics
        metrics_history["train_loss"].append(train_loss)
        metrics_history["mAP"].append(metrics["mAP"])
        metrics_history["mAP_50"].append(metrics["mAP_50"])
        metrics_history["mAP_75"].append(metrics["mAP_75"])
        metrics_history["mAP_small"].append(metrics["mAP_small"])
        metrics_history["mAP_medium"].append(metrics["mAP_medium"])
        metrics_history["mAP_large"].append(metrics["mAP_large"])
        
        if scheduler is not None:
            scheduler.step()

        if early_stopping:
            current_map = metrics["mAP"]  # or another metric you deem fit
            if current_map > best_val_map:
                best_val_map = current_map
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= 5:
                print("Early stopping triggered!")
                break

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, mAP: {metrics['mAP']:.4f}")
        print(metrics)

        # Save metrics to file for later visualization
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_history, f)

    return train_losses, val_metrics, metrics_file_path