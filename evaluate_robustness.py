import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from models import UNet
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from skimage.util import random_noise


class RobustnessDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), perturbation=None, level=0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.img_files = sorted(os.listdir(img_dir))
        self.perturbation = perturbation
        self.level = level

        # Basic transforms
        self.resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.mask_resize = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()

    def apply_perturbation(self, image):
        if self.perturbation is None:
            return image

        img_np = np.array(image)

        if self.perturbation == 'gaussian_noise':
            # Gaussian noise with increasing std
            std = self.level * 2
            noise = np.random.normal(0, std, img_np.shape)
            perturbed = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        elif self.perturbation == 'gaussian_blur':
            # Gaussian blur with increasing kernel size
            sigma = (self.level + 1) * 0.5
            perturbed = gaussian_filter(img_np, sigma=(sigma, sigma, 0))
            perturbed = perturbed.astype(np.uint8)

        elif self.perturbation == 'contrast_increase':
            # Contrast increase
            factor = 1.0 + self.level * 0.05
            perturbed = np.clip(img_np * factor, 0, 255).astype(np.uint8)

        elif self.perturbation == 'contrast_decrease':
            # Contrast decrease
            factor = 1.0 - self.level * 0.1
            perturbed = np.clip(img_np * factor, 0, 255).astype(np.uint8)

        elif self.perturbation == 'brightness_increase':
            # Brightness increase
            increase = self.level * 5
            perturbed = np.clip(img_np + increase, 0, 255).astype(np.uint8)

        elif self.perturbation == 'brightness_decrease':
            # Brightness decrease
            decrease = self.level * 5
            perturbed = np.clip(img_np - decrease, 0, 255).astype(np.uint8)

        elif self.perturbation == 'occlusion':
            # Random square occlusion
            h, w = img_np.shape[:2]
            size = (self.level + 1) * 5
            y = np.random.randint(0, h - size)
            x = np.random.randint(0, w - size)
            perturbed = img_np.copy()
            perturbed[y:y + size, x:x + size] = 0

        elif self.perturbation == 'salt_pepper':
            # Salt and pepper noise
            amount = self.level * 0.02
            perturbed = random_noise(img_np, mode='s&p', amount=amount)
            perturbed = np.clip(perturbed * 255, 0, 255).astype(np.uint8)

        else:
            return image

        return Image.fromarray(perturbed)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.img_files[idx].replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply resize
        image = self.resize(image)
        mask = self.mask_resize(mask)

        # Apply perturbation
        image = self.apply_perturbation(image)

        # Convert to tensor
        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))

        # Map mask values: 0->0 (background), 1->1 (cat), 255->2 (dog)
        mask = torch.where(mask == 255, torch.tensor(2), mask)
        mask = mask.long()

        return {'image': image, 'mask': mask}


def evaluate_model(model, dataloader, device):
    model.eval()
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Calculate Dice score
            dice_scores = []
            for i in range(3):  # 3 classes
                pred_mask = (preds == i)
                true_mask = (masks == i)
                intersection = (pred_mask & true_mask).sum().float()
                union = pred_mask.sum() + true_mask.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                dice_scores.append(dice.item())

            total_dice += np.mean(dice_scores)
            num_samples += 1

    return total_dice / num_samples


def plot_robustness_results(results, perturbation_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(results, marker='o')
    plt.title(f'Segmentation Accuracy vs {perturbation_name}')
    plt.xlabel('Perturbation Level')
    plt.ylabel('Mean Dice Score')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def save_example_perturbations(dataset_class, img_dir, mask_dir, save_dir, perturbation):
    plt.figure(figsize=(15, 3))

    # Original image
    dataset = dataset_class(img_dir, mask_dir, perturbation=None)
    img = dataset[0]['image']
    plt.subplot(1, 5, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title('Original')
    plt.axis('off')

    # Perturbed images
    for i, level in enumerate([2, 4, 6, 8]):
        dataset = dataset_class(img_dir, mask_dir, perturbation=perturbation, level=level)
        img = dataset[0]['image']
        plt.subplot(1, 5, i + 2)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f'Level {level}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{perturbation}_examples.png'))
    plt.close()


def main():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'unet_best_model.pth'
    test_img_dir = os.path.join('Dataset', 'Test', 'color')
    test_mask_dir = os.path.join('Dataset', 'Test', 'label')
    results_dir = 'robustness_results'
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model = UNet(n_channels=3, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    perturbations = [
        ('gaussian_noise', 'Gaussian Noise', 10),
        ('gaussian_blur', 'Gaussian Blur', 10),
        ('contrast_increase', 'Contrast Increase', 10),
        ('contrast_decrease', 'Contrast Decrease', 10),
        ('brightness_increase', 'Brightness Increase', 10),
        ('brightness_decrease', 'Brightness Decrease', 10),
        ('occlusion', 'Occlusion', 10),
        ('salt_pepper', 'Salt & Pepper Noise', 10)
    ]

    # Evaluate each perturbation
    for perturbation, name, num_levels in perturbations:
        print(f"\nEvaluating {name}...")
        results = []

        # Save example perturbations
        save_example_perturbations(RobustnessDataset, test_img_dir, test_mask_dir,
                                   results_dir, perturbation)

        # Evaluate each level
        for level in tqdm(range(num_levels)):
            dataset = RobustnessDataset(
                test_img_dir,
                test_mask_dir,
                perturbation=perturbation,
                level=level
            )

            dataloader = DataLoader(
                dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            dice_score = evaluate_model(model, dataloader, device)
            results.append(dice_score)

        # Plot and save results
        plot_robustness_results(
            results,
            name,
            os.path.join(results_dir, f'{perturbation}_plot.png')
        )

        # Save numerical results
        np.save(os.path.join(results_dir, f'{perturbation}_results.npy'), results)


if __name__ == '__main__':
    main()