import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import cv2
from skimage.util import random_noise
from typing import List, Tuple, Dict
import json

from models import UNet
from dataset import PetSegmentationDataset


def apply_gaussian_noise(image: torch.Tensor, std: float) -> torch.Tensor:
    """Apply Gaussian noise with given standard deviation"""
    noise = torch.randn_like(image) * std
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


def apply_gaussian_blur(image: torch.Tensor, num_convolutions: int) -> torch.Tensor:
    """Apply Gaussian blur by convolving multiple times"""
    if num_convolutions == 0:
        return image

    # Convert to numpy for OpenCV processing
    img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    kernel_size = (3, 3)

    for _ in range(num_convolutions):
        img_np = cv2.GaussianBlur(img_np, kernel_size, 0)

    # Convert back to torch tensor
    return torch.from_numpy(img_np).float().permute(2, 0, 1) / 255


def apply_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image contrast"""
    return torch.clamp(image * factor, 0, 1)


def apply_brightness(image: torch.Tensor, value: float) -> torch.Tensor:
    """Adjust image brightness"""
    return torch.clamp(image + value / 255, 0, 1)


def apply_occlusion(image: torch.Tensor, size: int) -> torch.Tensor:
    """Apply random square occlusion"""
    if size == 0:
        return image

    occluded = image.clone()
    _, H, W = image.shape

    # Random position for occlusion
    x = np.random.randint(0, W - size)
    y = np.random.randint(0, H - size)

    occluded[:, y:y + size, x:x + size] = 0
    return occluded


def apply_salt_and_pepper(image: torch.Tensor, amount: float) -> torch.Tensor:
    """Apply salt and pepper noise"""
    # Convert to numpy for skimage processing
    img_np = image.permute(1, 2, 0).numpy()
    noisy = random_noise(img_np, mode='s&p', amount=amount)
    return torch.from_numpy(noisy).float().permute(2, 0, 1)


def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate mean Dice score across all classes"""
    dice_scores = []

    for class_idx in range(3):  # 3 classes: background, cat, dog
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)

        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum() + target_mask.sum()

        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())

    return np.mean(dice_scores)


def evaluate_robustness(
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        save_dir: str
) -> Dict:
    """Evaluate model robustness against different perturbations"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Define perturbation parameters
    perturbations = {
        'gaussian_noise': {'values': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 'func': apply_gaussian_noise},
        'gaussian_blur': {'values': list(range(10)), 'func': apply_gaussian_blur},
        'contrast_increase': {'values': [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
                              'func': apply_contrast},
        'contrast_decrease': {'values': [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
                              'func': apply_contrast},
        'brightness_increase': {'values': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45], 'func': apply_brightness},
        'brightness_decrease': {'values': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                                'func': lambda x, v: apply_brightness(x, -v)},
        'occlusion': {'values': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45], 'func': apply_occlusion},
        'salt_and_pepper': {'values': [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
                            'func': apply_salt_and_pepper}
    }

    results = {name: {'dice_scores': [], 'example_images': []} for name in perturbations.keys()}

    # Get one batch for visualization
    example_batch = next(iter(test_loader))
    example_image = example_batch['image'][0].to(device)
    example_mask = example_batch['mask'][0].to(device)

    with torch.no_grad():
        for pert_name, pert_info in tqdm(perturbations.items(), desc="Evaluating perturbations"):
            print(f"\nEvaluating {pert_name}")

            # Save example perturbed images
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'Examples of {pert_name}')

            for idx, value in enumerate(pert_info['values']):
                # Process entire test set for each perturbation level
                batch_dice_scores = []

                for batch in test_loader:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                    # Apply perturbation
                    perturbed_images = torch.stack([
                        pert_info['func'](img, value) for img in images
                    ]).to(device)

                    # Get predictions
                    outputs = model(perturbed_images)
                    preds = torch.argmax(outputs, dim=1)

                    # Calculate Dice score
                    batch_dice = calculate_dice_score(preds, masks)
                    batch_dice_scores.append(batch_dice)

                # Record mean Dice score for this perturbation level
                mean_dice = np.mean(batch_dice_scores)
                results[pert_name]['dice_scores'].append(mean_dice)

                # Save example image for visualization
                if idx < 10:  # Save first 10 examples
                    perturbed_example = pert_info['func'](example_image, value)
                    ax = axes[idx // 5, idx % 5]
                    ax.imshow(perturbed_example.cpu().permute(1, 2, 0))
                    ax.set_title(f'Value: {value:.2f}\nDice: {mean_dice:.3f}')
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{pert_name}_examples.png'))
            plt.close()

            # Plot Dice scores
            plt.figure(figsize=(10, 6))
            plt.plot(pert_info['values'], results[pert_name]['dice_scores'], '-o')
            plt.title(f'Mean Dice Score vs {pert_name}')
            plt.xlabel('Perturbation Value')
            plt.ylabel('Mean Dice Score')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'{pert_name}_plot.png'))
            plt.close()

    # Save numerical results
    with open(os.path.join(save_dir, 'robustness_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return results


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = 'robustness_results'
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = UNet(n_channels=3, n_classes=3).to(device)
    checkpoint = torch.load('/home/yashagarwal/CV_Yash/results/unet2/best_model.pth')  # Update path as needed
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test dataset and loader
    test_dataset = PetSegmentationDataset(
        root_dir='./Dataset',  # Update path as needed
        split='test',
        img_size=(256, 256),
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Run robustness evaluation
    results = evaluate_robustness(model, test_loader, device, save_dir)

    print("\nRobustness evaluation completed!")
    print(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    main()