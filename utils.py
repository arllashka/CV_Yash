import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Optional
import json
from datetime import datetime


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


def save_predictions(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        save_dir: str,
        num_samples: int = 5
):
    """Save model predictions as images"""
    model.eval()
    plots_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(plots_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            masks = batch['mask'].cpu()
            filenames = batch['filename']

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            for j in range(len(images)):
                plt.figure(figsize=(15, 5))

                # Plot original image
                plt.subplot(1, 3, 1)
                plt.imshow(images[j].cpu().permute(1, 2, 0))
                plt.title('Input Image')
                plt.axis('off')

                # Plot ground truth
                plt.subplot(1, 3, 2)
                plt.imshow(masks[j], cmap='tab10', vmin=0, vmax=2)
                plt.title('Ground Truth')
                plt.axis('off')

                # Plot prediction
                plt.subplot(1, 3, 3)
                plt.imshow(preds[j], cmap='tab10', vmin=0, vmax=2)
                plt.title('Prediction')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'pred_{filenames[j]}.png'))
                plt.close()


def evaluate_and_save_metrics(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        save_dir: str
):
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