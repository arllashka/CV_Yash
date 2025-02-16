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
    
    def generate_prompt_point(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a random point within the foreground object"""
        # Get indices of non-background pixels
        fg_indices = torch.nonzero(mask > 0)
        if len(fg_indices) == 0:
            # If no foreground, return center point
            h, w = mask.shape
            return torch.tensor([h//2, w//2]), mask[h//2, w//2]
        
        # Randomly select one point
        idx = torch.randint(len(fg_indices), (1,))
        point = fg_indices[idx].squeeze()
        class_id = mask[point[0], point[1]]
        
        return point, class_id
    
    def create_point_heatmap(self, point: torch.Tensor) -> torch.Tensor:
        """Create Gaussian heatmap around the prompt point"""
        h, w = self.img_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        # Create 2D Gaussian
        heatmap = torch.exp(-((x - point[1])**2 + (y - point[0])**2) / (2 * 10**2))
        return heatmap.unsqueeze(0)  # Add channel dimension
    
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
        mask = torch.where(mask == 255, torch.tensor(2), mask)
        
        # Convert mask to Long type (int64)
        mask = mask.long()  # Add this line
        
        if self.prompt_based:
            point, class_id = self.generate_prompt_point(mask)
            heatmap = self.create_point_heatmap(point)
            return {
                'image': image,
                'mask': mask,
                'point': point,
                'point_class': class_id,
                'point_heatmap': heatmap
            }
        
        return {
            'image': image,
            'mask': mask
        }