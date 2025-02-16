import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from datetime import datetime

from dataset import PetSegmentationDataset
from models import UNet
from trainer import SegmentationTrainer
from utils import plot_training_history, save_predictions, evaluate_and_save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for Pet Segmentation')
    parser.add_argument('--data_root', type=str, default='./Dataset',
                        help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of worker threads for data loading')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping')
    return parser.parse_args()


def create_dataloaders(args):
    # Create datasets
    train_dataset = PetSegmentationDataset(
        args.data_root,
        split='train',
        img_size=(256, 256),
        augment=True
    )

    val_dataset = PetSegmentationDataset(
        args.data_root,
        split='val',
        img_size=(256, 256),
        augment=False
    )

    test_dataset = PetSegmentationDataset(
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
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def main():
    args = parse_args()

    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'unet_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(args)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Create model
    model = UNet(n_channels=3, n_classes=3).to(device)
    model._init_weights()

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = SegmentationTrainer(
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

    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_dir)

    # Load best model for evaluation
    best_model_path = os.path.join(save_dir, 'models', 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save predictions
    print("Saving example predictions...")
    save_predictions(model, test_loader, device, save_dir, num_samples=5)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_and_save_metrics(model, test_loader, device, save_dir)

    # Print final metrics
    print("\nTest Set Results:")
    print(f"Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {test_metrics['mean_dice']:.4f}")
    print("\nPer-class IoU:")
    for class_name in ['background', 'cat', 'dog']:
        print(f"{class_name}: {test_metrics[f'{class_name}_iou']:.4f}")


if __name__ == '__main__':
    main()