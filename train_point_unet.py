import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

from point_unet import PointUNet
from point_dataset import PointSegmentationDataset
from trainer import SegmentationTrainer
from utils import plot_training_history, save_predictions, evaluate_and_save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train Point-based UNet for Pet Segmentation')
    parser.add_argument('--data_root', type=str, default='./Dataset',
                        help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping')
    return parser.parse_args()


class PointSegmentationTrainer(SegmentationTrainer):
    """Modified trainer for point-based segmentation"""

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_metrics = {}

        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                point_heatmaps = batch['point_heatmap'].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images, point_heatmaps)
                loss = self.criterion(outputs, masks)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)

                # Update totals
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'mean_iou': metrics['mean_iou']})

        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_metrics = {}

        with tqdm(val_loader, desc='Validation') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                point_heatmaps = batch['point_heatmap'].to(self.device)

                # Forward pass
                outputs = self.model(images, point_heatmaps)
                loss = self.criterion(outputs, masks)

                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)

                # Update totals
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v

                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'mean_iou': metrics['mean_iou']})

        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: v / len(val_loader) for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss

        return avg_metrics


def save_point_predictions(model, test_loader, device, save_dir, num_samples=10):
    """Save predictions with point visualization"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    cat_predictions = []  # (IoU score, image, mask, pred, point, filename)
    dog_predictions = []  # (IoU score, image, mask, pred, point, filename)

    with torch.no_grad():
        for batch in test_loader:
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
            plt.plot(point[1], point[0], 'rx', markersize=10)  # Add red X for point
            plt.title('Input Image with Point')
            plt.axis('off')

            # Point heatmap
            plt.subplot(1, 4, 2)
            heatmap = torch.zeros_like(mask, dtype=torch.float32)
            y, x = point
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


def main():
    args = parse_args()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'point_unet_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = PointSegmentationDataset(
        args.data_root,
        split='train',
        img_size=(256, 256),
        augment=True
    )

    val_dataset = PointSegmentationDataset(
        args.data_root,
        split='val',
        img_size=(256, 256),
        augment=False
    )

    test_dataset = PointSegmentationDataset(
        args.data_root,
        split='test',
        img_size=(256, 256),
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model and training components
    model = PointUNet(n_channels=3, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create trainer
    trainer = PointSegmentationTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        save_dir=save_dir
    )

    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience
    )

    # Load best model for evaluation
    best_model_path = os.path.join(save_dir, 'models', 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Calculate and save final metrics
    print("\nCalculating final metrics...")
    test_metrics = evaluate_and_save_metrics(model, test_loader, device, save_dir)

    # Save predictions with point visualization
    print("\nSaving example predictions...")
    save_point_predictions(model, test_loader, device, save_dir, num_samples=10)

    # Print final results
    print("\nTraining complete! Final metrics:")
    print(f"Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {test_metrics['mean_dice']:.4f}")
    print("\nPer-class IoU:")
    for class_name in ['background', 'cat', 'dog']:
        print(f"{class_name}: {test_metrics[f'{class_name}_iou']:.4f}")
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()