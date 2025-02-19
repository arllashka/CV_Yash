import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np


class PointSegmentationDataset(Dataset):
    """Dataset class for point-based segmentation"""

    def __init__(
            self,
            root_dir,
            split='train',
            img_size=(256, 256),
            augment=True
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        # Setup paths
        if split == 'train' or split == 'val':
            base_dir = os.path.join(root_dir, 'TrainVal')
        else:
            base_dir = os.path.join(root_dir, 'Test')

        self.img_dir = os.path.join(base_dir, 'color')
        self.mask_dir = os.path.join(base_dir, 'label')
        self.img_files = sorted(os.listdir(self.img_dir))

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

    def generate_point_heatmap(self, point):
        """Generate Gaussian heatmap for the given point"""
        y, x = point
        heatmap = np.zeros(self.img_size, dtype=np.float32)

        # Generate 2D Gaussian
        sigma = min(self.img_size) / 16  # Adaptive sigma based on image size
        y_grid, x_grid = np.ogrid[:self.img_size[0], :self.img_size[1]]
        heatmap = np.exp(-((y_grid - y) ** 2 + (x_grid - x) ** 2) / (2 * sigma ** 2))

        return torch.FloatTensor(heatmap).unsqueeze(0)

    def sample_point(self, mask, target_class):
        """Sample a random point from the target class region"""
        valid_points = torch.nonzero(mask == target_class)
        if len(valid_points) == 0:
            return None

        # Randomly select one point
        idx = torch.randint(len(valid_points), (1,))[0]
        return valid_points[idx].tolist()

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

        # Sample a point from either cat or dog region
        target_class = torch.randint(1, 3, (1,)).item()  # 1 for cat, 2 for dog
        point = self.sample_point(mask, target_class)

        # If no valid points found for the target class, try the other class
        if point is None:
            target_class = 3 - target_class  # Switch between 1 and 2
            point = self.sample_point(mask, target_class)

        # If still no valid points, use center point
        if point is None:
            point = [self.img_size[0] // 2, self.img_size[1] // 2]

        # Generate point heatmap
        point_heatmap = self.generate_point_heatmap(point)

        return {
            'image': image,
            'mask': mask,
            'point': point,
            'point_heatmap': point_heatmap,
            'filename': self.img_files[idx]
        }