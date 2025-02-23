import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Optional
import json
from datetime import datetime
from point_unet import PointUNet
from models import PointUNet


def save_plot(
        data: Dict[str, list],
        title: str,
        xlabel: str,
        ylabel: str,
        save_path: str,
        legend_labels: Optional[list] = None
):
    """Save a line plot to disk"""
    plt.figure(figsize=(10, 6))
    for i, (key, values) in enumerate(data.items()):
        label = legend_labels[i] if legend_labels else key
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history: Dict[str, list], save_dir: str):
    """Plot and save training history"""
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot loss
    save_plot(
        data={'train': history['train_loss'], 'val': history['val_loss']},
        title='Training and Validation Loss',
        xlabel='Epoch',
        ylabel='Loss',
        save_path=os.path.join(plots_dir, f'loss_plot_{timestamp}.png'),
        legend_labels=['Training', 'Validation']
    )

    # Plot IoU
    save_plot(
        data={'train': history['train_mean_iou'], 'val': history['val_mean_iou']},
        title='Training and Validation Mean IoU',
        xlabel='Epoch',
        ylabel='Mean IoU',
        save_path=os.path.join(plots_dir, f'iou_plot_{timestamp}.png'),
        legend_labels=['Training', 'Validation']
    )


def evaluate_and_save_metrics(model, dataloader, device, save_dir):
    """Evaluate model and save detailed metrics"""
    model.eval()
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    class_metrics = {
        'background': {'iou': [], 'dice': []},
        'cat': {'iou': [], 'dice': []},
        'dog': {'iou': [], 'dice': []}
    }

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Check if model requires point heatmap
            if isinstance(model, PointUNet):
                point_heatmaps = batch['point_heatmap'].to(device)
                outputs = model(images, point_heatmaps)
            else:
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            # Calculate metrics for each class
            for class_idx, class_name in enumerate(['background', 'cat', 'dog']):
                pred_mask = (preds == class_idx)
                target_mask = (masks == class_idx)

                intersection = (pred_mask & target_mask).sum().float()
                union = (pred_mask | target_mask).sum().float()

                iou = (intersection + 1e-6) / (union + 1e-6)
                dice = (2 * intersection + 1e-6) / (pred_mask.sum() + target_mask.sum() + 1e-6)

                class_metrics[class_name]['iou'].append(iou.item())
                class_metrics[class_name]['dice'].append(dice.item())

    # Calculate final metrics
    final_metrics = {}
    for class_name in class_metrics:
        final_metrics[f'{class_name}_iou'] = np.mean(class_metrics[class_name]['iou'])
        final_metrics[f'{class_name}_dice'] = np.mean(class_metrics[class_name]['dice'])

    final_metrics['mean_iou'] = np.mean([
        final_metrics[f'{class_name}_iou']
        for class_name in ['background', 'cat', 'dog']
    ])
    final_metrics['mean_dice'] = np.mean([
        final_metrics[f'{class_name}_dice']
        for class_name in ['background', 'cat', 'dog']
    ])

    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_path = os.path.join(metrics_dir, f'test_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    return final_metrics


def save_predictions(model, test_loader, device, save_dir, num_samples=10):
    """Save best predictions for both cats and dogs"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    cat_predictions = []  # (IoU score, image, mask, pred, filename, point_heatmap)
    dog_predictions = []  # (IoU score, image, mask, pred, filename, point_heatmap)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']

            # Handle point-based models
            if isinstance(model, PointUNet):
                point_heatmaps = batch['point_heatmap'].to(device)
                outputs = model(images, point_heatmaps)
            else:
                outputs = model(images)
                point_heatmaps = None

            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each image
            for idx, (image, mask, pred, filename) in enumerate(zip(images, masks, preds, filenames)):
                point_heatmap = point_heatmaps[idx] if point_heatmaps is not None else None

                # For cats (class 1)
                if 1 in mask:
                    cat_mask = (mask == 1)
                    cat_pred = (pred == 1)
                    intersection = torch.logical_and(cat_mask, cat_pred).sum().float()
                    union = torch.logical_or(cat_mask, cat_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    cat_predictions.append((iou, image, mask, pred, filename, point_heatmap))

                # For dogs (class 2)
                if 2 in mask:
                    dog_mask = (mask == 2)
                    dog_pred = (pred == 2)
                    intersection = torch.logical_and(dog_mask, dog_pred).sum().float()
                    union = torch.logical_or(dog_mask, dog_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    dog_predictions.append((iou, image, mask, pred, filename, point_heatmap))

        # Sort by IoU and get top num_samples
        cat_predictions.sort(key=lambda x: x[0], reverse=True)
        dog_predictions.sort(key=lambda x: x[0], reverse=True)

        cat_samples = cat_predictions[:num_samples]
        dog_samples = dog_predictions[:num_samples]

        print(f"\nFound {len(cat_predictions)} cat images and {len(dog_predictions)} dog images")
        if cat_samples:
            print(f"Best cat IoU: {cat_samples[0][0]:.4f}")
        if dog_samples:
            print(f"Best dog IoU: {dog_samples[0][0]:.4f}")

        def save_prediction(sample, prefix):
            iou, image, mask, pred, filename, point_heatmap = sample

            if isinstance(model, PointUNet):
                plt.figure(figsize=(20, 5))

                plt.subplot(1, 4, 1)
                plt.imshow(image.cpu().permute(1, 2, 0))
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(point_heatmap.cpu().squeeze(), cmap='viridis')
                plt.title('Point Heatmap')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(mask.cpu(), cmap='tab10', vmin=0, vmax=2)
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(pred.cpu(), cmap='tab10', vmin=0, vmax=2)
                plt.title(f'Prediction (IoU: {iou:.4f})')
                plt.axis('off')
            else:
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(image.cpu().permute(1, 2, 0))
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(mask.cpu(), cmap='tab10', vmin=0, vmax=2)
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred.cpu(), cmap='tab10', vmin=0, vmax=2)
                plt.title(f'Prediction (IoU: {iou:.4f})')
                plt.axis('off')

            plt.savefig(os.path.join(save_dir, 'predictions', f'{prefix}_iou{iou:.4f}_{filename}.png'))
            plt.close()

        # Save top predictions for each class
        print("\nSaving top cat predictions...")
        for idx, sample in enumerate(cat_samples):
            save_prediction(sample, f'cat_{idx + 1}')

        print("\nSaving top dog predictions...")
        for idx, sample in enumerate(dog_samples):
            save_prediction(sample, f'dog_{idx + 1}')