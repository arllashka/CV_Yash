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

    # Evaluate model
    print("\nCalculating metrics...")
    metrics = evaluate_model(model, test_loader, device)

    # Save metrics
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f'test_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Print results
    print("\nEvaluation Results:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print("\nPer-class IoU:")
    for class_name in ['background', 'cat', 'dog']:
        print(f"{class_name}: {metrics[f'{class_name}_iou']:.4f}")

if __name__ == '__main__':
    main()