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

if __name__ == '__main__':
    main()