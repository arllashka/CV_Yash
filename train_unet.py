import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from dataset import PetSegmentationDataset
from models import UNet
from trainer import SegmentationTrainer
from utils import plot_training_history, save_predictions, evaluate_and_save_metrics


def main():
    # Create save directory
    save_dir = './final_results'
    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = PetSegmentationDataset(
        './Dataset',
        split='train',
        img_size=(256, 256),
        augment=True
    )

    val_dataset = PetSegmentationDataset(
        './Dataset',
        split='val',
        img_size=(256, 256),
        augment=False
    )

    test_dataset = PetSegmentationDataset(
        './Dataset',
        split='test',
        img_size=(256, 256),
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Create model and training components
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
        num_epochs=50,
        early_stopping_patience=10
    )

    # Save only the best validation loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'val_loss.png'))
    plt.close()

    # Load best model for evaluation
    best_model_path = os.path.join(save_dir, 'models', 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate and save best predictions for cats and dogs
    print("\nFinding best predictions for cats and dogs...")
    model.eval()
    with torch.no_grad():
        cat_predictions = []  # (IoU score, image, mask, pred, filename)
        dog_predictions = []

        for batch in tqdm(test_loader, desc="Evaluating predictions"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each image
            for idx, (image, mask, pred, filename) in enumerate(zip(images, masks, preds, filenames)):
                # For cats (class 1)
                if 1 in mask:
                    cat_mask = (mask == 1)
                    cat_pred = (pred == 1)
                    intersection = torch.logical_and(cat_mask, cat_pred).sum().float()
                    union = torch.logical_or(cat_mask, cat_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    cat_predictions.append((iou, image, mask, pred, filename))

                # For dogs (class 2)
                if 2 in mask:
                    dog_mask = (mask == 2)
                    dog_pred = (pred == 2)
                    intersection = torch.logical_and(dog_mask, dog_pred).sum().float()
                    union = torch.logical_or(dog_mask, dog_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    dog_predictions.append((iou, image, mask, pred, filename))

        # Sort by IoU and get top 10
        cat_predictions.sort(key=lambda x: x[0], reverse=True)
        dog_predictions.sort(key=lambda x: x[0], reverse=True)

        cat_samples = cat_predictions[:10]
        dog_samples = dog_predictions[:10]

        print(f"\nFound {len(cat_predictions)} cat images and {len(dog_predictions)} dog images")
        print(f"Best cat IoU: {cat_samples[0][0]:.4f}")
        print(f"Best dog IoU: {dog_samples[0][0]:.4f}")

        # Save predictions for both cats and dogs
        os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

        def save_prediction(sample, prefix):
            iou, image, mask, pred, filename = sample

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

        print("\nSaving top 10 predictions for each class...")
        for idx, sample in enumerate(cat_samples):
            save_prediction(sample, f'cat_{idx + 1}')

        for idx, sample in enumerate(dog_samples):
            save_prediction(sample, f'dog_{idx + 1}')

    # Calculate and save final metrics
    print("\nCalculating final metrics...")
    test_metrics = evaluate_and_save_metrics(model, test_loader, device, save_dir)

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