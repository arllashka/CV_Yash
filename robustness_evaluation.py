import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PetSegmentationDataset
from models import UNet
from utils import evaluate_and_save_metrics
from torch.serialization import safe_globals, add_safe_globals

# Add numpy scalar to safe globals for loading model
add_safe_globals(['numpy.core.multiarray.scalar'])

# Configuration
model_path = 'results/unet2/best_model.pth'  # Replace with your actual model path
data_root = "./Dataset"  # Adjust this path
save_dir = "./robustness_results"
os.makedirs(save_dir, exist_ok=True)

# Load the trained model
model = UNet(n_channels=3, n_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load test dataset
test_dataset = PetSegmentationDataset(data_root, split='test', img_size=(256, 256), augment=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


class RobustnessEvaluator:
    @staticmethod
    def add_gaussian_noise(image, std):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)

    @staticmethod
    def apply_gaussian_blur(image, kernel_size):
        """Apply Gaussian blur"""
        if kernel_size <= 1:
            return image
        return F.gaussian_blur(image, kernel_size=(kernel_size, kernel_size))

    @staticmethod
    def adjust_contrast(image, factor):
        """Adjust image contrast"""
        return torch.clamp(factor * (image - 0.5) + 0.5, 0, 1)

    @staticmethod
    def adjust_brightness(image, delta):
        """Adjust image brightness"""
        return torch.clamp(image + delta, 0, 1)

    @staticmethod
    def apply_occlusion(image, size):
        """Apply random square occlusion"""
        if size <= 0:
            return image
        b, c, h, w = image.shape
        occluded = image.clone()

        # Calculate random position for occlusion
        x = torch.randint(0, w - size, (1,))
        y = torch.randint(0, h - size, (1,))

        # Apply occlusion
        occluded[:, :, y:y + size, x:x + size] = 0
        return occluded

    @staticmethod
    def add_salt_and_pepper(image, amount):
        """Add salt and pepper noise"""
        if amount <= 0:
            return image

        noise_mask = torch.rand_like(image)
        salt = (noise_mask < amount / 2).float()
        pepper = ((noise_mask >= amount / 2) & (noise_mask < amount)).float()

        noisy_image = image.clone()
        noisy_image[salt == 1] = 1
        noisy_image[pepper == 1] = 0
        return noisy_image


# Define perturbation configurations
perturbations = {
    "Gaussian Noise": {
        "levels": [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
        "function": RobustnessEvaluator.add_gaussian_noise
    },
    "Gaussian Blur": {
        "levels": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        "function": RobustnessEvaluator.apply_gaussian_blur
    },
    "Contrast Increase": {
        "levels": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
        "function": RobustnessEvaluator.adjust_contrast
    },
    "Contrast Decrease": {
        "levels": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
        "function": RobustnessEvaluator.adjust_contrast
    },
    "Brightness Increase": {
        "levels": [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        "function": RobustnessEvaluator.adjust_brightness
    },
    "Brightness Decrease": {
        "levels": [0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45],
        "function": RobustnessEvaluator.adjust_brightness
    },
    "Occlusion": {
        "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        "function": RobustnessEvaluator.apply_occlusion
    },
    "Salt and Pepper": {
        "levels": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
        "function": RobustnessEvaluator.add_salt_and_pepper
    }
}


def evaluate_robustness():
    """Evaluate model robustness against different perturbations"""
    results = {}

    for perturb_type, config in perturbations.items():
        print(f"\nEvaluating {perturb_type}...")
        perturb_function = config["function"]
        dice_scores = []

        # Create example images for visualization
        example_images = []

        for level in config["levels"]:
            batch_dice_scores = []

            for batch_idx, batch in enumerate(test_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # Apply perturbation
                perturbed_images = perturb_function(images, level)

                # Save first batch's first image as example
                if batch_idx == 0:
                    example_images.append(perturbed_images[0].cpu())

                # Get predictions
                with torch.no_grad():
                    outputs = model(perturbed_images)
                    preds = torch.argmax(outputs, dim=1)

                # Calculate Dice score for each class
                for class_idx in range(3):  # background, cat, dog
                    pred_mask = (preds == class_idx)
                    target_mask = (masks == class_idx)

                    intersection = (pred_mask & target_mask).sum().float()
                    union = pred_mask.sum() + target_mask.sum()

                    if union > 0:
                        dice = (2 * intersection) / (union + 1e-6)
                        batch_dice_scores.append(dice.item())

            # Average Dice score for this perturbation level
            avg_dice = np.mean(batch_dice_scores)
            dice_scores.append(avg_dice)
            print(f"Level {level}: Mean Dice = {avg_dice:.4f}")

        results[perturb_type] = {
            "scores": dice_scores,
            "levels": config["levels"],
            "examples": example_images
        }

        # Plot and save example images
        plot_example_perturbations(perturb_type, example_images, config["levels"])

    # Plot overall results
    plot_robustness_results(results)

    return results


def plot_example_perturbations(perturb_type, images, levels):
    """Plot example images for each perturbation level"""
    n_examples = len(images)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    for i in range(min(n_examples, 10)):
        ax = axes[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f'Level: {levels[i]}')
        ax.axis('off')

    plt.suptitle(f'{perturb_type} Examples')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{perturb_type.lower().replace(" ", "_")}_examples.png'))
    plt.close()


def plot_robustness_results(results):
    """Plot robustness evaluation results"""
    plt.figure(figsize=(12, 8))

    for perturb_type, data in results.items():
        plt.plot(data["levels"], data["scores"], marker='o', label=perturb_type)

    plt.xlabel('Perturbation Level')
    plt.ylabel('Mean Dice Score')
    plt.title('Model Robustness Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'robustness_analysis.png'))
    plt.close()


if __name__ == "__main__":
    results = evaluate_robustness()

    # Save numerical results
    with open(os.path.join(save_dir, 'robustness_results.txt'), 'w') as f:
        for perturb_type, data in results.items():
            f.write(f"\n{perturb_type}:\n")
            for level, score in zip(data["levels"], data["scores"]):
                f.write(f"Level {level}: {score:.4f}\n")