import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as T
from skimage.util import random_noise
import cv2

from dataset import PetSegmentationDataset
from models import UNet
from utils import evaluate_and_save_metrics
from trainer import SegmentationTrainer


class RobustnessEvaluator:
    def __init__(self, model_path, data_root, device='cuda', batch_size=8):
        self.device = device
        self.data_root = data_root
        self.batch_size = batch_size

        # Load model
        self.model = UNet(n_channels=3, n_classes=3).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)  # Set weights_only to False
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Create test dataset and loader
        self.test_dataset = PetSegmentationDataset(
            data_root,
            split='test',
            img_size=(256, 256),
            augment=False
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Define perturbation parameters
        self.gaussian_noise_std = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.gaussian_blur_times = list(range(10))  # 0 to 9 times
        self.contrast_increase = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
        self.contrast_decrease = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
        self.brightness_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        self.occlusion_sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        self.salt_pepper_amounts = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]

    def apply_gaussian_noise(self, images, std):
        """Apply Gaussian noise to batch of images"""
        noise = torch.randn_like(images) * std
        noisy_images = images + noise
        return torch.clamp(noisy_images, 0, 1)

    def apply_gaussian_blur(self, images, times):
        """Apply Gaussian blur multiple times"""
        if times == 0:
            return images

        blurred = images.clone()
        kernel_size = 3
        sigma = 1.0

        for _ in range(times):
            blurred = torch.nn.functional.gaussian_blur(
                blurred,
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma)
            )
        return blurred

    def apply_contrast(self, images, factor):
        """Adjust contrast of images"""
        return torch.clamp(images * factor, 0, 1)

    def apply_brightness(self, images, value):
        """Adjust brightness of images"""
        return torch.clamp(images + value / 255.0, 0, 1)

    def apply_occlusion(self, images, size):
        """Apply random square occlusion"""
        if size == 0:
            return images

        occluded = images.clone()
        b, c, h, w = images.shape

        for i in range(b):
            # Random position for occlusion
            x = torch.randint(0, w - size, (1,))
            y = torch.randint(0, h - size, (1,))

            occluded[i, :, y:y + size, x:x + size] = 0

        return occluded

    def apply_salt_pepper(self, images, amount):
        """Apply salt and pepper noise"""
        if amount == 0:
            return images

        noisy = images.clone()

        for i in range(len(images)):
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            noisy_np = random_noise(img_np, mode='s&p', amount=amount)
            noisy[i] = torch.from_numpy(noisy_np.transpose(2, 0, 1)).to(images.device)

        return noisy.float()

    def calculate_dice_score(self, pred, target):
        """Calculate Dice score for batch"""
        pred = pred.argmax(dim=1)
        dice_scores = []

        for class_idx in range(3):  # 3 classes
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)

            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum()

            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.item())

        return np.mean(dice_scores)

    def evaluate_perturbation(self, perturb_func, values, name):
        """Evaluate model performance under specific perturbation"""
        print(f"\nEvaluating {name} perturbation...")
        mean_dice_scores = []

        for val in tqdm(values):
            batch_dice_scores = []

            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # Apply perturbation
                perturbed_images = perturb_func(images, val)

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(perturbed_images)

                # Calculate Dice score
                dice_score = self.calculate_dice_score(outputs, masks)
                batch_dice_scores.append(dice_score)

            mean_dice = np.mean(batch_dice_scores)
            mean_dice_scores.append(mean_dice)
            print(f"{name} level {val}: Mean Dice = {mean_dice:.4f}")

        return mean_dice_scores

    def save_example_perturbations(self, save_dir):
        """Save examples of each perturbation type"""
        os.makedirs(save_dir, exist_ok=True)

        # Get a single batch
        batch = next(iter(self.test_loader))
        image = batch['image'][0].to(self.device)  # Take first image

        perturbations = {
            'gaussian_noise': (self.apply_gaussian_noise, self.gaussian_noise_std[5]),
            'gaussian_blur': (self.apply_gaussian_blur, self.gaussian_blur_times[5]),
            'contrast_increase': (self.apply_contrast, self.contrast_increase[5]),
            'contrast_decrease': (self.apply_contrast, self.contrast_decrease[5]),
            'brightness_increase': (self.apply_brightness, self.brightness_values[5]),
            'brightness_decrease': (self.apply_brightness, -self.brightness_values[5]),
            'occlusion': (self.apply_occlusion, self.occlusion_sizes[5]),
            'salt_pepper': (self.apply_salt_pepper, self.salt_pepper_amounts[5])
        }

        # Original image
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 3, 1)
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.title('Original')
        plt.axis('off')

        for idx, (name, (func, val)) in enumerate(perturbations.items(), 2):
            plt.subplot(3, 3, idx)
            perturbed = func(image.unsqueeze(0), val)
            plt.imshow(perturbed[0].cpu().permute(1, 2, 0))
            plt.title(f'{name}\n(level={val})')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'perturbation_examples.png'))
        plt.close()

    def plot_results(self, results, save_dir):
        """Plot results for all perturbations"""
        os.makedirs(save_dir, exist_ok=True)

        for name, (values, scores) in results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values, scores, marker='o')
            plt.title(f'Model Performance Under {name} Perturbation')
            plt.xlabel('Perturbation Level')
            plt.ylabel('Mean Dice Score')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'{name}_plot.png'))
            plt.close()

    def run_full_evaluation(self, save_dir):
        """Run complete robustness evaluation"""
        os.makedirs(save_dir, exist_ok=True)

        results = {}

        # Evaluate all perturbation types
        results['Gaussian Noise'] = (
            self.gaussian_noise_std,
            self.evaluate_perturbation(self.apply_gaussian_noise, self.gaussian_noise_std, "Gaussian Noise")
        )

        results['Gaussian Blur'] = (
            self.gaussian_blur_times,
            self.evaluate_perturbation(self.apply_gaussian_blur, self.gaussian_blur_times, "Gaussian Blur")
        )

        results['Contrast Increase'] = (
            self.contrast_increase,
            self.evaluate_perturbation(self.apply_contrast, self.contrast_increase, "Contrast Increase")
        )

        results['Contrast Decrease'] = (
            self.contrast_decrease,
            self.evaluate_perturbation(self.apply_contrast, self.contrast_decrease, "Contrast Decrease")
        )

        results['Brightness Increase'] = (
            self.brightness_values,
            self.evaluate_perturbation(self.apply_brightness, self.brightness_values, "Brightness Increase")
        )

        results['Brightness Decrease'] = (
            self.brightness_values,
            self.evaluate_perturbation(
                lambda x, v: self.apply_brightness(x, -v),
                self.brightness_values,
                "Brightness Decrease"
            )
        )

        results['Occlusion'] = (
            self.occlusion_sizes,
            self.evaluate_perturbation(self.apply_occlusion, self.occlusion_sizes, "Occlusion")
        )

        results['Salt & Pepper'] = (
            self.salt_pepper_amounts,
            self.evaluate_perturbation(self.apply_salt_pepper, self.salt_pepper_amounts, "Salt & Pepper")
        )

        # Save example perturbations
        self.save_example_perturbations(save_dir)

        # Plot results
        self.plot_results(results, save_dir)

        # Save numerical results
        with open(os.path.join(save_dir, 'numerical_results.txt'), 'w') as f:
            for name, (values, scores) in results.items():
                f.write(f"\n{name} Results:\n")
                for val, score in zip(values, scores):
                    f.write(f"Level {val}: {score:.4f}\n")


if __name__ == "__main__":
    # Setup paths
    model_path = "/Users/yashagarwal/Desktop/CV_Yash/unet_best_model.pth"  # Updated correct path
    data_root = "./Dataset"
    save_dir = "./robustness_results"

    # Automatically detect if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create evaluator
    evaluator = RobustnessEvaluator(
        model_path=model_path,
        data_root=data_root,
        device=device,
        batch_size=8
    )

    # Run evaluation
    evaluator.run_full_evaluation(save_dir)