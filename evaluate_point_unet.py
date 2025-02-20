import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import numpy as np

from point_unet import PointUNet
from point_dataset import PointSegmentationDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Point-based UNet')
    parser.add_argument('--data_root', type=str, default='./Dataset',
                        help='path to dataset')
    parser.add_argument('--model_dir', type=str,
                        default='/home/yashagarwal/CV_Yash/results/point_unet_20250219_185721',
                        help='directory containing the model')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for evaluation')
    return parser.parse_args()


def save_point_predictions(model, test_loader, device, save_dir, num_samples=10):
    """Save predictions with point visualization"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    cat_predictions = []  # (IoU score, image, mask, pred, point, filename)
    dog_predictions = []  # (IoU score, image, mask, pred, point, filename)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            points = batch['point']
            point_heatmaps = batch['point_heatmap'].to(device)
            filenames = batch['filename']

            outputs = model(images, point_heatmaps)
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each image
            for idx, (image, mask, pred, point, filename) in enumerate(zip(images, masks, preds, points, filenames)):
                # For cats (class 1)
                if 1 in mask:
                    cat_mask = (mask == 1)
                    cat_pred = (pred == 1)
                    intersection = torch.logical_and(cat_mask, cat_pred).sum().float()
                    union = torch.logical_or(cat_mask, cat_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    cat_predictions.append((iou, image, mask, pred, point, filename))

                # For dogs (class 2)
                if 2 in mask:
                    dog_mask = (mask == 2)
                    dog_pred = (pred == 2)
                    intersection = torch.logical_and(dog_mask, dog_pred).sum().float()
                    union = torch.logical_or(dog_mask, dog_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    dog_predictions.append((iou, image, mask, pred, point, filename))

        # Sort by IoU and get top predictions
        cat_predictions.sort(key=lambda x: x[0], reverse=True)
        dog_predictions.sort(key=lambda x: x[0], reverse=True)

        cat_samples = cat_predictions[:num_samples]
        dog_samples = dog_predictions[:num_samples]

        def save_prediction(sample, prefix):
            iou, image, mask, pred, point, filename = sample

            plt.figure(figsize=(20, 5))

            # Original image with point
            plt.subplot(1, 4, 1)
            img_np = image.cpu().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            # Debug print to see point structure
            print(f"Point structure: {point}")
            y, x = point[0], point[1]  # Modified this line
            plt.plot(x, y, 'rx', markersize=10)  # Add red X for point
            plt.title('Input Image with Point')
            plt.axis('off')

            # Point heatmap
            plt.subplot(1, 4, 2)
            heatmap = torch.zeros_like(mask, dtype=torch.float32)
            sigma = min(image.shape[1:]) / 16
            y_grid, x_grid = torch.meshgrid(torch.arange(image.shape[1]), torch.arange(image.shape[2]))
            heatmap = torch.exp(-((y_grid - y) ** 2 + (x_grid - x) ** 2) / (2 * sigma ** 2))
            plt.imshow(heatmap.cpu(), cmap='hot')
            plt.title('Point Heatmap')
            plt.axis('off')

            # Ground truth
            plt.subplot(1, 4, 3)
            plt.imshow(mask.cpu(), cmap='tab10', vmin=0, vmax=2)
            plt.title('Ground Truth')
            plt.axis('off')

            # Prediction
            plt.subplot(1, 4, 4)
            plt.imshow(pred.cpu(), cmap='tab10', vmin=0, vmax=2)
            plt.title(f'Prediction (IoU: {iou:.4f})')
            plt.axis('off')

            plt.savefig(os.path.join(save_dir, 'predictions', f'{prefix}_iou{iou:.4f}_{filename}.png'))
            plt.close()

        # Save predictions
        print("\nSaving predictions...")
        for idx, sample in enumerate(cat_samples):
            save_prediction(sample, f'cat_{idx + 1}')

        for idx, sample in enumerate(dog_samples):
            save_prediction(sample, f'dog_{idx + 1}')


def evaluate_model(model, dataloader, device):
    """Evaluate model metrics"""
    model.eval()
    class_metrics = {
        'background': {'iou': [], 'dice': []},
        'cat': {'iou': [], 'dice': []},
        'dog': {'iou': [], 'dice': []}
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            point_heatmaps = batch['point_heatmap'].to(device)

            outputs = model(images, point_heatmaps)
            preds = torch.argmax(outputs, dim=1)

            for class_idx, class_name in enumerate(['background', 'cat', 'dog']):
                pred_mask = (preds == class_idx)
                target_mask = (masks == class_idx)

                intersection = (pred_mask & target_mask).sum().float()
                union = (pred_mask | target_mask).sum().float()

                iou = (intersection + 1e-6) / (union + 1e-6)
                dice = (2 * intersection + 1e-6) / (pred_mask.sum() + target_mask.sum() + 1e-6)

                class_metrics[class_name]['iou'].append(iou.item())
                class_metrics[class_name]['dice'].append(dice.item())

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

    return final_metrics


def main():
    args = parse_args()
    save_dir = args.model_dir

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test dataset
    test_dataset = PointSegmentationDataset(
        args.data_root,
        split='test',
        img_size=(256, 256),
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model and load checkpoint
    model = PointUNet(n_channels=3, n_classes=3).to(device)
    checkpoint_path = os.path.join(save_dir, 'models', 'best_model.pth')
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save predictions with visualization
    print("\nGenerating and saving test predictions...")
    save_point_predictions(model, test_loader, device, save_dir, num_samples=10)

    print("\nPredictions have been saved to:", os.path.join(save_dir, 'predictions'))


if __name__ == '__main__':
    main()