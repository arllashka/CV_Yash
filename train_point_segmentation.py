import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from point_dataset import PointPromptDataset
from point_model import PointSegmentationModel
from utils import save_predictions, evaluate_and_save_metrics


class PointSegmentationTrainer:
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            criterion: nn.Module = nn.BCELoss(),
            optimizer: torch.optim.Optimizer = None,
            save_dir: str = 'results'
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir

        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        num_batches = len(train_loader)

        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                point_heatmaps = batch['point_heatmap'].to(self.device)
                masks = batch['mask'].to(self.device).float()

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images, point_heatmaps)
                loss = self.criterion(outputs, masks)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate metrics
                pred_masks = (outputs > 0.5).float()
                intersection = (pred_masks * masks).sum((1, 2))
                union = (pred_masks + masks).bool().float().sum((1, 2))
                iou = (intersection / (union + 1e-6)).mean()
                dice = (2 * intersection / (pred_masks.sum((1, 2)) + masks.sum((1, 2)) + 1e-6)).mean()

                total_loss += loss.item()
                total_iou += iou.item()
                total_dice += dice.item()

                pbar.set_postfix({
                    'loss': loss.item(),
                    'iou': iou.item(),
                    'dice': dice.item()
                })

        metrics = {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches,
            'dice': total_dice / num_batches
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        num_batches = len(val_loader)

        cat_ious = []
        dog_ious = []

        for batch in val_loader:
            images = batch['image'].to(self.device)
            point_heatmaps = batch['point_heatmap'].to(self.device)
            masks = batch['mask'].to(self.device).float()
            class_indices = batch['class_idx']

            # Forward pass
            outputs = self.model(images, point_heatmaps)
            loss = self.criterion(outputs, masks)

            # Calculate metrics
            pred_masks = (outputs > 0.5).float()
            intersection = (pred_masks * masks).sum((1, 2))
            union = (pred_masks + masks).bool().float().sum((1, 2))
            iou = intersection / (union + 1e-6)
            dice = 2 * intersection / (pred_masks.sum((1, 2)) + masks.sum((1, 2)) + 1e-6)

            # Store class-specific IoUs
            for idx, class_idx in enumerate(class_indices):
                if class_idx == 1:  # Cat
                    cat_ious.append(iou[idx].item())
                elif class_idx == 2:  # Dog
                    dog_ious.append(iou[idx].item())

            total_loss += loss.item()
            total_iou += iou.mean().item()
            total_dice += dice.mean().item()

        # Calculate background IoU
        background_ious = []
        for batch in val_loader:
            images = batch['image'].to(self.device)
            point_heatmaps = batch['point_heatmap'].to(self.device)
            # Original full mask (with all classes)
            full_mask = batch['full_mask'].to(self.device)

            outputs = self.model(images, point_heatmaps)
            pred_masks = (outputs > 0.5).float()

            # Calculate background IoU (where full_mask == 0)
            background_mask = (full_mask == 0).float()
            background_pred = (pred_masks == 0).float()

            intersection = (background_pred * background_mask).sum((1, 2))
            union = (background_pred + background_mask).bool().float().sum((1, 2))
            background_iou = (intersection / (union + 1e-6))
            background_ious.extend(background_iou.cpu().tolist())

        metrics = {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches,
            'dice': total_dice / num_batches,
            'background_iou': sum(background_ious) / len(background_ious) if background_ious else 0,
            'cat_iou': sum(cat_ious) / len(cat_ious) if cat_ious else 0,
            'dog_iou': sum(dog_ious) / len(dog_ious) if dog_ious else 0,
            'mean_iou': (sum(background_ious) / len(background_ious) if background_ious else 0 +
                                                                                             sum(cat_ious) / len(
                cat_ious) if cat_ious else 0 +
                                           sum(dog_ious) / len(dog_ious) if dog_ious else 0) / 3
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }

        if is_best:
            path = os.path.join(self.save_dir, 'models', 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, 'models', f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)

    def save_example_predictions(self, val_loader: DataLoader, epoch: int, num_samples: int = 3):
        self.model.eval()
        os.makedirs(os.path.join(self.save_dir, 'predictions', f'epoch_{epoch}'), exist_ok=True)

        with torch.no_grad():
            # Collect examples for both classes
            cat_examples = []
            dog_examples = []
            needed_cats = num_samples
            needed_dogs = num_samples

            for batch in val_loader:
                if needed_cats == 0 and needed_dogs == 0:
                    break

                class_indices = batch['class_idx']

                # Find cat examples
                if needed_cats > 0 and 1 in class_indices:
                    cat_indices = (class_indices == 1).nonzero(as_tuple=True)[0]
                    for cat_idx in cat_indices[:needed_cats]:
                        cat_examples.append({
                            'image': batch['image'][cat_idx:cat_idx + 1],
                            'point_heatmap': batch['point_heatmap'][cat_idx:cat_idx + 1],
                            'mask': batch['mask'][cat_idx:cat_idx + 1],
                            'point': batch['point'][cat_idx:cat_idx + 1],
                            'filename': [batch['filename'][cat_idx]],
                            'class_idx': batch['class_idx'][cat_idx:cat_idx + 1]
                        })
                    needed_cats = max(0, needed_cats - len(cat_indices))

                # Find dog examples
                if needed_dogs > 0 and 2 in class_indices:
                    dog_indices = (class_indices == 2).nonzero(as_tuple=True)[0]
                    for dog_idx in dog_indices[:needed_dogs]:
                        dog_examples.append({
                            'image': batch['image'][dog_idx:dog_idx + 1],
                            'point_heatmap': batch['point_heatmap'][dog_idx:dog_idx + 1],
                            'mask': batch['mask'][dog_idx:dog_idx + 1],
                            'point': batch['point'][dog_idx:dog_idx + 1],
                            'filename': [batch['filename'][dog_idx]],
                            'class_idx': batch['class_idx'][dog_idx:dog_idx + 1]
                        })

            # Process and save predictions for both cats and dogs
            for examples, animal in [(cat_examples, 'cat'), (dog_examples, 'dog')]:
                if batch is None:
                    continue

                images = batch['image'].to(self.device)
                point_heatmaps = batch['point_heatmap'].to(self.device)
                masks = batch['mask'].to(self.device)
                points = batch['point']
                filenames = batch['filename']

                outputs = self.model(images, point_heatmaps)
                pred_masks = (outputs > 0.5).float()

                for idx in range(len(images)):
                    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

                    # Original image with point
                    img = images[idx].cpu().permute(1, 2, 0)
                    point_y, point_x = points[idx]
                    axs[0, 0].imshow(img)
                    axs[0, 0].plot(point_x, point_y, 'rx', markersize=10)
                    axs[0, 0].set_title(f'Input {animal.title()} Image with Point')
                    axs[0, 0].axis('off')

                    # Point heatmap
                    axs[0, 1].imshow(point_heatmaps[idx, 0].cpu(), cmap='hot')
                    axs[0, 1].set_title('Point Heatmap')
                    axs[0, 1].axis('off')

                    # Ground truth mask
                    axs[1, 0].imshow(masks[idx].cpu(), cmap='gray')
                    axs[1, 0].set_title('Ground Truth')
                    axs[1, 0].axis('off')

                    # Predicted mask
                    axs[1, 1].imshow(pred_masks[idx].cpu(), cmap='gray')
                    axs[1, 1].set_title('Prediction')
                    axs[1, 1].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        self.save_dir,
                        'predictions',
                        f'epoch_{epoch}',
                        f'{animal}_{filenames[idx]}'
                    ))
                    plt.close()

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            early_stopping_patience: int = 10
    ) -> dict:
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_iou'].append(train_metrics['iou'])

            # Validate
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_iou'].append(val_metrics['iou'])

            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train IoU: {train_metrics['iou']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
            print(f"Cat IoU: {val_metrics['cat_iou']:.4f}, Dog IoU: {val_metrics['dog_iou']:.4f}")

            # Save example predictions
            if epoch % 5 == 0:
                self.save_example_predictions(val_loader, epoch)

            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        return history


def main():
    parser = argparse.ArgumentParser(description='Train Point-based Segmentation Model')
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
    args = parser.parse_args()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'point_segmentation_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = PointPromptDataset(
        args.data_root,
        split='train',
        img_size=(256, 256),
        augment=True
    )

    val_dataset = PointPromptDataset(
        args.data_root,
        split='val',
        img_size=(256, 256),
        augment=False
    )

    test_dataset = PointPromptDataset(
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
    model = PointSegmentationModel().to(device)
    criterion = nn.BCELoss()
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

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("\nTest set metrics:")
    print(f"Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {test_metrics['mean_dice']:.4f}")
    print("\nPer-class metrics:")
    print(f"Background - IoU: {test_metrics['background_iou']:.4f}, Dice: {test_metrics['background_dice']:.4f}")
    print(f"Cat - IoU: {test_metrics['cat_iou']:.4f}, Dice: {test_metrics['cat_dice']:.4f}")
    print(f"Dog - IoU: {test_metrics['dog_iou']:.4f}, Dice: {test_metrics['dog_dice']:.4f}")

    # Save test metrics
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"\nTraining completed! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()