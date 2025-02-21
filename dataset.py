import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from typing import Optional, Tuple


class PetSegmentationDataset(Dataset):
    """Dataset class for Oxford-IIIT Pet Dataset segmentation"""

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            img_size: Tuple[int, int] = (256, 256),
            augment: bool = True,
            prompt_based: bool = False
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.prompt_based = prompt_based

        # Setup paths based on the actual dataset structure
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

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> dict:
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
            # Use same random seed for synchronized augmentation
            seed = torch.randint(2147483647, (1,))[0]
            torch.manual_seed(seed)
            image = self.aug_transforms(image)
            torch.manual_seed(seed)
            mask = self.mask_aug_transforms(mask)

        # Convert to tensor
        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))

        # Map mask values: 0->0 (background), 1->1 (cat), 255->2 (dog)
        if 1 in mask.unique().tolist():
            mask = torch.where(mask == 255, torch.tensor(1), mask)
        else:
            mask = torch.where(mask == 255, torch.tensor(2), mask)
        mask = mask.long()

        return {
            'image': image,
            'mask': mask,
            'filename': self.img_files[idx]  # Include filename for visualization
        }