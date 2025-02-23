import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime
from dataset import PetSegmentationDataset
from models import UNet


class RobustnessEvaluator:
    def __init__(self, model, device, save_dir="./robustness_results"):
        """Initialize robustness evaluator"""
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Create subdirectories
        self.plots_dir = os.path.join(save_dir, 'plots')
        self.examples_dir = os.path.join(save_dir, 'examples')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.examples_dir, exist_ok=True)

    def apply_gaussian_noise(self, images, std):
        """Add Gaussian noise with specified standard deviation"""
        noise = torch.randn_like(images) * std
        noisy_images = torch.clamp(images + noise, 0, 1)
        return noisy_images

    def apply_gaussian_blur(self, images, n_times):
        """Apply 3x3 Gaussian mask n times"""
        if n_times == 0:
            return images

        # Define 3x3 Gaussian kernel
        kernel = torch.tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ], device=images.device) / 16.0
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Apply convolution n times
        blurred = images
        for _ in range(n_times):
            padded = F.pad(blurred, (1, 1, 1, 1), mode='reflect')
            blurred = F.conv2d(padded, kernel, groups=3)
        return blurred

    def adjust_contrast(self, images, factor):
        """Adjust image contrast"""
        mean = torch.mean(images, dim=(2, 3), keepdim=True)
        adjusted = (images - mean) * factor + mean
        return torch.clamp(adjusted, 0, 1)

    def adjust_brightness(self, images, delta):
        """Adjust image brightness"""
        adjusted = images + delta / 255.0  # Convert from [0,255] to [0,1] range
        return torch.clamp(adjusted, 0, 1)

    def apply_occlusion(self, images, size):
        """Apply random square occlusion"""
        if size == 0:
            return images

        occluded = images.clone()
        b, c, h, w = images.shape

        # Ensure size doesn't exceed image dimensions
        size = min(size, min(h, w) - 1)

        # Random position for occlusion
        for i in range(b):
            x = torch.randint(0, w - size, (1,))
            y = torch.randint(0, h - size, (1,))
            occluded[i, :, y:y + size, x:x + size] = 0

        return occluded

    def add_salt_and_pepper(self, images, amount):
        """Add salt and pepper noise"""
        if amount == 0:
            return images

        noisy = images.clone()
        b, c, h, w = images.shape

        num_salt = int(amount * h * w / 2)
        num_pepper = int(amount * h * w / 2)

        for i in range(b):
            # Add salt
            salt_coords = [
                torch.randint(0, h, (num_salt,)),
                torch.randint(0, w, (num_salt,))
            ]
            noisy[i, :, salt_coords[0], salt_coords[1]] = 1

            # Add pepper
            pepper_coords = [
                torch.randint(0, h, (num_pepper,)),
                torch.randint(0, w, (num_pepper,))
            ]
            noisy[i, :, pepper_coords[0], pepper_coords[1]] = 0

        return noisy

    def calculate_dice_score(self, pred, target):
        """Calculate mean Dice score across all classes"""
        dice_scores = []

        for class_idx in range(3):  # background, cat, dog
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)

            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum()

            if union > 0:
                dice = (2 * intersection) / (union + 1e-6)
                dice_scores.append(dice.item())

        return np.mean(dice_scores)

    def evaluate_robustness(self, test_loader):
        """Evaluate model robustness against all perturbations"""
        perturbations = {
            "Gaussian Noise": {
                "func": self.apply_gaussian_noise,
                "levels": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
                "x_label": "Standard Deviation"
            },
            "Gaussian Blur": {
                "func": self.apply_gaussian_blur,
                "levels": list(range(10)),  # 0 to 9 convolutions
                "x_label": "Number of 3x3 Convolutions"
            },
            "Contrast Increase": {
                "func": self.adjust_contrast,
                "levels": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
                "x_label": "Contrast Factor"
            },
            "Contrast Decrease": {
                "func": self.adjust_contrast,
                "levels": [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
                "x_label": "Contrast Factor"
            },
            "Brightness Increase": {
                "func": self.adjust_brightness,
                "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "x_label": "Brightness Addition"
            },
            "Brightness Decrease": {
                "func": lambda x, l: self.adjust_brightness(x, -l),
                "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "x_label": "Brightness Subtraction"
            },
            "Occlusion": {
                "func": self.apply_occlusion,
                "levels": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "x_label": "Square Size"
            },
            "Salt and Pepper": {
                "func": self.add_salt_and_pepper,
                "levels": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
                "x_label": "Noise Amount"
            }
        }

        # Get example image for visualization
        example_batch = next(iter(test_loader))
        example_image = example_batch['image'][0:1].to(self.device)

        # Store results
        results = {}
        plt.figure(figsize=(15, 10))

        # Evaluate each perturbation
        for name, config in perturbations.items():
            print(f"\nEvaluating {name}...")
            perturb_func = config["func"]
            levels = config["levels"]
            scores = []
            examples = []

            # Test each perturbation level
            for level in tqdm(levels, desc=f"{name} levels"):
                batch_scores = []

                # Save example of perturbation
                perturbed_example = perturb_func(example_image, level)
                examples.append(perturbed_example[0].cpu())

                # Evaluate on test set
                for batch in test_loader:
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)

                    # Apply perturbation
                    perturbed = perturb_func(images, level)

                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(perturbed)
                        preds = torch.argmax(outputs, dim=1)

                    # Calculate Dice score
                    dice = self.calculate_dice_score(preds, masks)
                    batch_scores.append(dice)

                mean_dice = np.mean(batch_scores)
                scores.append(mean_dice)

            # Save results
            results[name] = {
                "levels": levels,
                "scores": scores
            }

            # Plot examples
            self.plot_examples(examples, levels, name)

            # Add to performance plot
            plt.plot(levels, scores, marker='o', label=name)

        # Finalize and save performance plot
        plt.xlabel('Perturbation Strength')
        plt.ylabel('Mean Dice Score')
        plt.title('Segmentation Performance vs. Perturbation Strength')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'robustness_analysis.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        return results

    def plot_examples(self, images, levels, perturb_name):
        """Plot grid of example perturbations"""
        n_examples = len(images)
        n_cols = 5
        n_rows = (n_examples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_examples):
            row, col = i // n_cols, i % n_cols
            axes[row, col].imshow(images[i].permute(1, 2, 0))
            axes[row, col].set_title(f'Level: {levels[i]}')
            axes[row, col].axis('off')

        # Clear empty subplots
        for i in range(n_examples, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')

        plt.suptitle(f'{perturb_name} Examples', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.examples_dir, f'{perturb_name.lower().replace(" ", "_")}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()


def main():
    # Configuration
    model_path = 'results/unet2/best_model.pth'  # Your model path
    data_root = './Dataset'  # Your dataset path
    save_dir = './robustness_results'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test dataset
    test_dataset = PetSegmentationDataset(
        data_root,
        split='test',
        img_size=(256, 256),
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = UNet(n_channels=3, n_classes=3)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)
    model.eval()

    # Create evaluator and run evaluation
    evaluator = RobustnessEvaluator(model, device, save_dir)
    results = evaluator.evaluate_robustness(test_loader)

    # Save numerical results
    with open(os.path.join(save_dir, 'robustness_results.txt'), 'w') as f:
        f.write("Robustness Evaluation Results\n")
        f.write("============================\n\n")

        for perturb_type, data in results.items():
            f.write(f"{perturb_type}:\n")
            f.write("-" * len(perturb_type) + "\n")
            f.write("Level\tDice Score\n")
            for level, score in zip(data["levels"], data["scores"]):
                f.write(f"{level:.4f}\t{score:.4f}\n")

            # Calculate statistics
            baseline = data["scores"][0]
            worst = min(data["scores"])
            max_degradation = baseline - worst
            worst_level = data["levels"][np.argmin(data["scores"])]

            f.write(f"\nStatistics:\n")
            f.write(f"Baseline score: {baseline:.4f}\n")
            f.write(f"Worst score: {worst:.4f}\n")
            f.write(f"Maximum degradation: {max_degradation:.4f}\n")
            f.write(f"Most vulnerable at level: {worst_level}\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"\nEvaluation complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()