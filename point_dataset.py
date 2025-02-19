import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple
import random


class PointPromptDataset(Dataset):
    """Dataset class for point-prompted segmentation"""

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            img_size: Tuple[int, int] = (256, 256),
            augment: bool = True,
            sigma: float = 10.0  # Gaussian sigma for point heatmap
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.sigma = sigma

        # Setup paths based on the actual dataset structure
        if split == 'train' or split == 'val':
            base_dir = os.path.join(root_dir, 'TrainVal')
        else:
            base_dir = os.path.join(root_dir, 'Test')

        self.img_dir = os.path.join(base_dir, 'color')
        self.mask_dir = os.path.join(base_dir, 'label')

        # Create separate lists for cat and dog images
        self.cat_images = []
        self.dog_images = []

        # Categorize images based on content
        print(f"Categorizing {split} images...")
        for img_file in sorted(os.listdir(self.img_dir)):
            mask_file = os.path.join(self.mask_dir, img_file.replace('.jpg', '.png'))
            mask = np.array(Image.open(mask_file))

            # Check for cat (class 1) and dog (class 2)
            if 1 in mask:
                self.cat_images.append(img_file)
            if 2 in mask or 255 in mask:  # 255 is mapped to 2 for dogs
                self.dog_images.append(img_file)

        print(f"Found {len(self.cat_images)} cat images and {len(self.dog_images)} dog images")

        # For training/validation, we'll sample equally from both classes
        self.max_samples = max(len(self.cat_images), len(self.dog_images))

        # Basic transforms
        self.resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
        self.mask_resize = T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()

        # Augmentation transforms
        if self.augment:
            self.aug_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=T.InterpolationMode.BILINEAR
                )
            ])
            self.mask_aug_transforms = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
                T.RandomResizedCrop(
                    img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=T.InterpolationMode.NEAREST
                )
            ])

    def generate_point_heatmap(self, mask: torch.Tensor, point: Tuple[int, int]) -> torch.Tensor:
        """Generate Gaussian heatmap for a given point"""
        y, x = point
        heatmap = torch.zeros(self.img_size)

        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.img_size[0], dtype=torch.float32),
            torch.arange(self.img_size[1], dtype=torch.float32)
        )

        # Calculate Gaussian heatmap
        heatmap = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * self.sigma ** 2))
        return heatmap

    def sample_point(self, mask: torch.Tensor, class_idx: int) -> Optional[Tuple[int, int]]:
        """Sample a random point from the region of the specified class"""
        # Get coordinates where mask equals class_idx
        y_coords, x_coords = torch.where(mask == class_idx)

        if len(y_coords) == 0:
            return None

        # Randomly select one point
        idx = random.randint(0, len(y_coords) - 1)
        return (y_coords[idx].item(), x_coords[idx].item())

    def __len__(self) -> int:
        # Return twice the max length to ensure equal sampling from both classes
        return 2 * self.max_samples

    def __getitem__(self, idx: int) -> dict:
        # Determine if we're sampling a cat or dog image
        is_cat = idx < self.max_samples

        # Get the appropriate image list and class index
        if is_cat:
            img_list = self.cat_images
            class_idx = 1  # cat
            actual_idx = idx % len(self.cat_images)
        else:
            img_list = self.dog_images
            class_idx = 2  # dog
            actual_idx = idx % len(self.dog_images)

        # Load image and mask
        img_file = img_list[actual_idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # Apply resize
        image = self.resize(image)
        mask = self.mask_resize(mask)

        # Apply augmentations if enabled
        if self.augment:
            seed = torch.randint(2147483647, (1,))[0]
            torch.manual_seed(seed)
            image = self.aug_transforms(image)
            torch.manual_seed(seed)
            mask = self.mask_aug_transforms(mask)

        # Convert to tensor
        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))

        # Map mask values: 0->0 (background), 1->1 (cat), 255->2 (dog)
        mask = torch.where(mask == 255, torch.tensor(2), mask)
        mask = mask.long()

        # Sample point from the correct class
        point = self.sample_point(mask, class_idx)
        if point is None:
            # This shouldn't happen given our categorization, but just in case
            raise RuntimeError(f"Could not find point for class {class_idx} in image {img_file}")

        # Generate point heatmap
        point_heatmap = self.generate_point_heatmap(mask, point)

        # Create binary mask for the selected class
        target_mask = (mask == class_idx).long()

        return {
            'image': image,
            'point_heatmap': point_heatmap.unsqueeze(0),  # Add channel dimension
            'mask': target_mask,
            'point': torch.tensor(point),
            'class_idx': torch.tensor(class_idx),
            'filename': img_file
        }