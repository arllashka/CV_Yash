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

# Configuration
model_path = 'results/unet2/best_model.pth'  # Replace with your actual model path
data_root = "./Dataset"  # Adjust this path
save_dir = "./robustness_results"
os.makedirs(save_dir, exist_ok=True)

# Load the trained model
model = UNet(n_channels=3, n_classes=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # First attempt: try loading with default settings
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
except Exception as e1:
    try:
        # Second attempt: try loading with weights_only=False
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e2:
        # Third attempt: try loading state dict directly
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
        except Exception as e3:
            raise Exception(f"Failed to load model. Tried multiple methods:\n1. {str(e1)}\n2. {str(e2)}\n3. {str(e3)}")
model.to(device)
model.eval()

# Load test dataset
test_dataset = PetSegmentationDataset(data_root, split='test', img_size=(256, 256), augment=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


class RobustnessEvaluator:
    @staticmethod
    def add_gaussian_noise(image, std):
        """Add Gaussian noise to image with specified standard deviation"""
        # Scale image to 0-255 range
        image_255 = image * 255
        noise = torch.randn_like(image_255) * std
        noisy = torch.clamp(image_255 + noise, 0, 255)
        return noisy / 255

    @staticmethod
    def apply_gaussian_blur(image, n_times):
        """Apply 3x3 Gaussian blur n times"""
        if n_times == 0:
            return image

        # Apply 3x3 Gaussian blur n_times
        blurred = image
        kernel_size = 3
        for _ in range(n_times):
            blurred = F.gaussian_blur(blurred, kernel_size=(kernel_size, kernel_size))
        return blurred

    @staticmethod
    def adjust_contrast(image, factor):
        """Adjust image contrast by multiplying with factor"""
        # Scale to 0-255, adjust contrast, then back to 0-1
        image_255 = image * 255
        adjusted = torch.clamp(factor * image_255, 0, 255)
        return adjusted / 255

    @staticmethod
    def adjust_brightness(image, delta):
        """Adjust image brightness by adding/subtracting delta"""
        # Scale to 0-255, adjust brightness, then back to 0-1
        image_255 = image * 255
        if isinstance(delta, (int, float)) and delta >= 0:
            # For brightness increase
            adjusted = torch.clamp(image_255 + delta, 0, 255)
        else:
            # For brightness decrease
            adjusted = torch.clamp(image_255 - delta, 0, 255)
        return adjusted / 255

    @staticmethod
    def apply_occlusion(image, size):
        """Apply random square occlusion with given edge length"""
        if size <= 0:
            return image

        b, c, h, w = image.shape
        occluded = image.clone()

        if size >= min(h, w):
            size = min(h, w) - 1

        # Calculate random position for occlusion
        x = torch.randint(0, w - size, (1,))
        y = torch.randint(0, h - size, (1,))

        # Apply black occlusion
        occluded[:, :, y:y + size, x:x + size] = 0
        return occluded

    @staticmethod
    def add_salt_and_pepper(image, amount):
        """Add salt and pepper noise with specified amount"""
        if amount <= 0:
            return image

        noise_mask = torch.rand_like(image)
        salt = (noise_mask < amount / 2).float()
        pepper = ((noise_mask >= amount / 2) & (noise_mask < amount)).float()

        noisy_image = image.clone()
        noisy_image[salt == 1] = 1  # White noise (255 in 0-255 scale)
        noisy_image[pepper == 1] = 0  # Black noise (


# Define perturbation configurations exactly as specified in the assignment
perturbations = {
    "Gaussian Noise": {
        "levels": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],  # Standard deviations as specified
        "function": RobustnessEvaluator.add_gaussian_noise
    },
    "Gaussian Blur": {
        "levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Number of times to apply 3x3 mask
        "function": RobustnessEvaluator.apply_gaussian_blur
    },
    "Contrast Increase": {
        "levels": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],  # Multiplication factors
        "function": RobustnessEvaluator.adjust_contrast
    },
    "Contrast Decrease": {
        "levels": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],  # Multiplication factors
        "function": RobustnessEvaluator.adjust_contrast
    },
    "Brightness Increase": {
        "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # Additive values
        "function": RobustnessEvaluator.adjust_brightness
    },
    "Brightness Decrease": {
        "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # Subtractive values
        "function": RobustnessEvaluator.adjust_brightness
    },
    "Occlusion": {
        "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],  # Square edge lengths
        "function": RobustnessEvaluator.apply_occlusion
    },
    "Salt and Pepper": {
        "levels": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],  # Noise amounts
        "function": RobustnessEvaluator.add_salt_and_pepper
    }
}


def evaluate_robustness():
    """Evaluate model robustness against different perturbations"""
    results = {}

    # Create directories for saving results
    examples_dir = os.path.join(save_dir, 'perturbation_examples')
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize figure for all robustness plots
    plt.figure(figsize=(15, 10))

    for perturb_type, config in perturbations.items():
        print(f"\nEvaluating {perturb_type}...")
        perturb_function = config["function"]
        dice_scores = []

        # Save example perturbed images
        example_batch = next(iter(test_loader))
        example_image = example_batch['image'][0:1].to(device)  # Take first image

        perturbed_examples = []
        for level in config["levels"]:
            batch_dice_scores = []

            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # Apply perturbation
                perturbed_images = perturb_function(images, level)

                # Get predictions
                with torch.no_grad():
                    outputs = model(perturbed_images)
                    preds = torch.argmax(outputs, dim=1)

                # Calculate Dice score
                dice_score = calculate_dice_score(preds, masks)
                batch_dice_scores.append(dice_score)

            # Save example of perturbation
            if len(perturbed_examples) < len(config["levels"]):
                perturbed = perturb_function(example_image, level)
                perturbed_examples.append(perturbed[0].cpu())

            # Average Dice score for this perturbation level
            avg_dice = np.mean(batch_dice_scores)
            dice_scores.append(avg_dice)
            print(f"Level {level}: Mean Dice = {avg_dice:.4f}")

        results[perturb_type] = {
            "scores": dice_scores,
            "levels": config["levels"]
        }

        # Plot and save example perturbations
        save_perturbation_examples(
            perturb_type,
            perturbed_examples,
            config["levels"],
            examples_dir
        )

        # Plot robustness curve for this perturbation
        plt.plot(config["levels"], dice_scores, marker='o', label=perturb_type)

    # Finalize and save overall robustness plot
    plt.xlabel('Perturbation Strength')
    plt.ylabel('Mean Dice Score')
    plt.title('Segmentation Performance vs. Perturbation Strength')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save numerical results
    with open(os.path.join(save_dir, 'robustness_results.txt'), 'w') as f:
        for perturb_type, data in results.items():
            f.write(f"\n{perturb_type}:\n")
            f.write("Level\tDice Score\n")
            for level, score in zip(data["levels"], data["scores"]):
                f.write(f"{level}\t{score:.4f}\n")

    return results


def save_perturbation_examples(perturb_type, examples, levels, save_dir):
    """Save example images for each perturbation level"""
    n_examples = len(examples)
    n_cols = 5
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_examples:
            img = examples[i].permute(1, 2, 0).numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title(f'Level: {levels[i]}')
        ax.axis('off')

    plt.suptitle(f'{perturb_type} Examples', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{perturb_type.lower().replace(" ", "_")}_examples.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def calculate_dice_score(pred, target):
    """Calculate Dice score for multi-class segmentation"""
    num_classes = 3  # background, cat, dog
    dice_scores = []

    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)

        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum() + target_mask.sum()

        if union > 0:
            dice = (2 * intersection) / (union + 1e-6)
            dice_scores.append(dice.item())

    return np.mean(dice_scores)


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