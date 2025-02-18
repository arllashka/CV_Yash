import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

from dataset import PetSegmentationDataset
from models import Autoencoder
from utils import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description='Train Autoencoder for Pet Segmentation')
    parser.add_argument('--data_root', type=str, default='./Dataset',
                        help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping')
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            images = batch['image'].to(device)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    for batch in val_loader:
        images = batch['image'].to(device)
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    args = parse_args()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'autoencoder_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'autoencoder'), exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets and dataloaders
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

    # Create model and training components
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print("Starting autoencoder pre-training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, 'autoencoder', 'best_autoencoder.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Plot and save training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'autoencoder_loss.png'))
    plt.close()

    print(f"\nAutoencoder pre-training completed! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()