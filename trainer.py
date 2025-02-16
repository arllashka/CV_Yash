import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=1e-4)
        self.lr_scheduler = lr_scheduler
        
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate IoU and Dice coefficient for each class"""
        pred_class = torch.argmax(pred, dim=1)
        metrics = {}
        
        # Calculate for each class
        for class_idx in range(pred.shape[1]):  # number of classes
            pred_mask = (pred_class == class_idx)
            target_mask = (target == class_idx)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            # IoU
            iou = (intersection + 1e-6) / (union + 1e-6)
            metrics[f'iou_class_{class_idx}'] = iou.item()
            
            # Dice coefficient
            dice = (2 * intersection + 1e-6) / (pred_mask.sum() + target_mask.sum() + 1e-6)
            metrics[f'dice_class_{class_idx}'] = dice.item()
        
        # Calculate mean metrics
        metrics['mean_iou'] = np.mean([metrics[f'iou_class_{i}'] 
                                     for i in range(pred.shape[1])])
        metrics['mean_dice'] = np.mean([metrics[f'dice_class_{i}'] 
                                      for i in range(pred.shape[1])])
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {}
        
        with tqdm(train_loader, desc='Training') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                
                # Update totals
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'mean_iou': metrics['mean_iou']})
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        total_metrics = {}
        
        with tqdm(val_loader, desc='Validation') as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                
                # Update totals
                total_loss += loss.item()
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'mean_iou': metrics['mean_iou']})
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: v / len(val_loader) for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
    ) -> Dict[str, List[float]]:
        """Training loop with early stopping"""
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mean_iou': [], 'val_mean_iou': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_mean_iou'].append(train_metrics['mean_iou'])
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_mean_iou'].append(val_metrics['mean_iou'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train mIoU: {train_metrics['mean_iou']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val mIoU: {val_metrics['mean_iou']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Learning rate scheduling
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_metrics['loss'])
        
        return history