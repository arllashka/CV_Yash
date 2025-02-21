import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
import json
from datetime import datetime

from models import UNet
from dataset import PetSegmentationDataset
from utils import evaluate_and_save_metrics


class RobustnessEvaluator:
    def __init__(self, model_path, data_root, save_dir):
        """
        Initialize the robustness evaluator with automatic device selection

        Args:
            model_path: Path to the best model checkpoint
            data_root: Root directory of the dataset
            save_dir: Directory to save results
        """
        # Automatically determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Create save directory
        self.save_dir = os.path.join(save_dir, 'robustness_results')
        os.makedirs(self.save_dir, exist_ok=True)

        # Load model
        try:
            print("Loading model...")
            self.model = UNet(n_channels=3, n_classes=3).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Create test dataset and loader
        try:
            print("Setting up dataset and dataloader...")
            self.test_dataset = PetSegmentationDataset(
                data_root,
                split='test',
                img_size=(256, 256),
                augment=False
            )

            # Adjust batch size and workers based on device
            batch_size = 16 if self.device.type == 'cuda' else 4
            cpu_count = os.cpu_count()
            num_workers = min(4, cpu_count - 1) if cpu_count else 0

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            print(f"Dataset setup complete. Using batch size: {batch_size}")
        except Exception as e:
            print(f"Error setting up dataset/dataloader: {str(e)}")
            raise

    def apply_gaussian_noise(self, images, std):
        """Apply Gaussian noise with given standard deviation"""
        noise = torch.randn_like(images) * std / 255.0
        noisy_images = images + noise
        return torch.clamp(noisy_images, 0, 1)

    def apply_gaussian_blur(self, images, iterations):
        """Apply Gaussian blur by convolving multiple times"""
        blurred = images.cpu().numpy()
        for _ in range(iterations):
            blurred = gaussian_filter(blurred, sigma=1)
        return torch.from_numpy(blurred).to(self.device)

    def adjust_contrast(self, images, factor):
        """Adjust image contrast"""
        adjusted = images * factor
        return torch.clamp(adjusted, 0, 1)

    def adjust_brightness(self, images, value):
        """Adjust image brightness"""
        adjusted = images + value / 255.0
        return torch.clamp(adjusted, 0, 1)

    def apply_occlusion(self, images, size):
        """Apply random square occlusion"""
        if size == 0:
            return images

        occluded = images.clone()
        for idx in range(images.shape[0]):
            if size > 0:
                h, w = images.shape[2], images.shape[3]
                top = np.random.randint(0, h - size) if h > size else 0
                left = np.random.randint(0, w - size) if w > size else 0
                occluded[idx, :, top:top + size, left:left + size] = 0

        return occluded

    def apply_salt_pepper(self, images, amount):
        """Apply salt and pepper noise"""
        noisy = images.cpu().numpy()
        for idx in range(noisy.shape[0]):
            noisy[idx] = random_noise(noisy[idx], mode='s&p', amount=amount)
        return torch.from_numpy(noisy).to(self.device)

    def evaluate_batch(self, images, masks):
        """Evaluate a batch and return Dice score"""
        with torch.no_grad():
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)

            dice_scores = []
            for pred, mask in zip(preds, masks):
                dice_per_class = []
                for class_idx in range(3):  # background, cat, dog
                    pred_mask = (pred == class_idx)
                    true_mask = (mask == class_idx)

                    intersection = (pred_mask & true_mask).sum().float()
                    union = pred_mask.sum() + true_mask.sum()

                    dice = (2 * intersection + 1e-6) / (union + 1e-6)
                    dice_per_class.append(dice.item())

                dice_scores.append(np.mean(dice_per_class))

            return np.mean(dice_scores)

    def evaluate_perturbation(self, perturbation_fn, param_range, param_name):
        """Evaluate model performance across a range of perturbation parameters"""
        scores = []
        examples = []

        for param in tqdm(param_range, desc=f'Evaluating {param_name}'):
            batch_scores = []

            for batch_idx, batch in enumerate(self.test_loader):
                try:
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)

                    # Apply perturbation
                    perturbed_images = perturbation_fn(images, param)

                    # Save example from first batch
                    if batch_idx == 0 and len(examples) < len(param_range):
                        examples.append(perturbed_images[0].cpu())

                    # Calculate score
                    batch_score = self.evaluate_batch(perturbed_images, masks)
                    batch_scores.append(batch_score)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nWARNING: GPU out of memory. Clearing cache and continuing...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise e

            scores.append(np.mean(batch_scores))

        return np.array(scores), examples

    def plot_results(self, param_range, scores, examples, param_name, save_name):
        """Plot and save results"""
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot 1: Performance curve
        ax1.plot(param_range, scores, 'b-o')
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Mean Dice Score')
        ax1.set_title(f'Segmentation Performance vs {param_name}')
        ax1.grid(True)

        # Plot 2: Example perturbations
        num_examples = len(examples)
        for idx, img in enumerate(examples):
            plt.subplot(2, num_examples // 2 + 1, idx + num_examples // 2 + 2)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'{param_name}={param_range[idx]}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{save_name}.png'))
        plt.close()

    def run_all_evaluations(self):
        """Run all robustness evaluations"""
        print("\nStarting robustness evaluations...")

        # Clear GPU cache if available
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        evaluation_params = [
            ('Gaussian Noise', self.apply_gaussian_noise,
             [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 'gaussian_noise'),

            ('Gaussian Blur', self.apply_gaussian_blur,
             list(range(10)), 'gaussian_blur'),

            ('Contrast Increase', self.adjust_contrast,
             [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
             'contrast_increase'),

            ('Contrast Decrease', self.adjust_contrast,
             [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
             'contrast_decrease'),

            ('Brightness Increase', self.adjust_brightness,
             [0, 5, 10, 15, 20, 25, 30, 35, 40, 45], 'brightness_increase'),

            ('Brightness Decrease', self.adjust_brightness,
             [0, -5, -10, -15, -20, -25, -30, -35, -40, -45], 'brightness_decrease'),

            ('Occlusion', self.apply_occlusion,
             [0, 5, 10, 15, 20, 25, 30, 35, 40, 45], 'occlusion'),

            ('Salt & Pepper', self.apply_salt_pepper,
             [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
             'salt_pepper')
        ]

        results = {}
        for name, fn, param_range, save_name in evaluation_params:
            print(f"\nEvaluating {name}...")
            try:
                scores, examples = self.evaluate_perturbation(fn, param_range, name)
                self.plot_results(param_range, scores, examples, name, save_name)
                results[name] = {
                    'params': param_range,
                    'scores': scores.tolist()
                }
                print(f"{name} evaluation completed successfully")
            except Exception as e:
                print(f"Error during {name} evaluation: {str(e)}")
                continue

        # Save numerical results
        with open(os.path.join(self.save_dir, 'numerical_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nAll evaluations completed. Results saved to {self.save_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the best model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')

    args = parser.parse_args()

    print("\nStarting Robustness Evaluation")
    print("=" * 50)
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Save directory: {args.save_dir}")

    try:
        evaluator = RobustnessEvaluator(
            model_path=args.model_path,
            data_root=args.data_root,
            save_dir=args.save_dir
        )
        evaluator.run_all_evaluations()

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise