import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from dataset import PetSegmentationDataset
from models import UNet
from trainer import SegmentationTrainer


class Config:
    # Data configs
    DATA_ROOT = './Dataset'
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    
    # Model configs
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 3  # background, cat, dog
    
    # Training configs
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    WEIGHT_DECAY = 1e-5


def create_dataloaders():
    """
    Create and return the train, validation, and test dataloaders.
    """
    # Create datasets
    train_dataset = PetSegmentationDataset(
        Config.DATA_ROOT,
        split='train',
        img_size=Config.IMG_SIZE,
        augment=True
    )
    
    val_dataset = PetSegmentationDataset(
        Config.DATA_ROOT,
        split='val',
        img_size=Config.IMG_SIZE,
        augment=False
    )
    
    test_dataset = PetSegmentationDataset(
        Config.DATA_ROOT,
        split='test',
        img_size=Config.IMG_SIZE,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def initialize_model():
    """
    Initialize the UNet model, optimizer, and loss criterion.
    """
    # Initialize model and move it to the appropriate device
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model._init_weights()
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def print_model_summary(model):
    """
    Print a summary of the model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def main():
    print(f"Using device: {Config.DEVICE}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Initialize model, optimizer, and criterion
    model, optimizer, criterion = initialize_model()
    
    # Print model summary
    print_model_summary(model)
    
    # Initialize trainer
    trainer = SegmentationTrainer(
        model=model,
        device=Config.DEVICE,
        criterion=criterion,
        optimizer=optimizer
    )
    
    # Start training
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=Config.NUM_EPOCHS,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Optionally, you can save the model or plot training history here
    # For example:
    # torch.save(model.state_dict(), "unet_model.pth")
    # plt.plot(history['train_loss'], label="Train Loss")
    # plt.plot(history['val_loss'], label="Val Loss")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()