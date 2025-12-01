"""
CNN Training Script
COMP64301: Computer Vision Coursework

This module handles training, validation, and evaluation of CNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import (
    set_seed, get_device, save_model, save_results,
    AverageMeter, create_experiment_dir
)


class CNNTrainer:
    """
    CNN Trainer class for CIFAR-10
    Handles training loop, validation, and metrics tracking
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        experiment_name=None
    ):
        """
        Initialize trainer
        
        Args:
            model: CNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration object
            experiment_name: Optional experiment name
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Setup device
        self.device = get_device()
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Create experiment directory
        if experiment_name is None:
            experiment_name = f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.exp_dir = create_experiment_dir(
            config.CNN_RESULTS_DIR,
            experiment_name
        )
        
        # Initialize tracking variables
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"\nTrainer initialized on device: {self.device}")
    
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return scheduler
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average loss, average accuracy)
        """
        self.model.train()
        
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('acc')
        
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.EPOCHS} [Train]'
        )
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = 100.0 * correct / targets.size(0)
            
            # Update meters
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(accuracy, inputs.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
        
        return loss_meter.avg, acc_meter.avg
    
    def validate(self, epoch):
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (average loss, average accuracy)
        """
        self.model.eval()
        
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('acc')
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f'Epoch {epoch}/{self.config.EPOCHS} [Val]'
            )
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                accuracy = 100.0 * correct / targets.size(0)
                
                # Update meters
                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update(accuracy, inputs.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.2f}%'
                })
        
        return loss_meter.avg, acc_meter.avg
    
    def test(self):
        """
        Test the model on test set
        
        Returns:
            tuple: (test loss, test accuracy)
        """
        self.model.eval()
        
        loss_meter = AverageMeter('loss')
        acc_meter = AverageMeter('acc')
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                accuracy = 100.0 * correct / targets.size(0)
                
                # Update meters
                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update(accuracy, inputs.size(0))
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'acc': f'{acc_meter.avg:.2f}%'
                })
        
        print(f"\nTest Results:")
        print(f"  Loss: {loss_meter.avg:.4f}")
        print(f"  Accuracy: {acc_meter.avg:.2f}%")
        
        return loss_meter.avg, acc_meter.avg
    
    def train(self):
        """
        Full training loop
        
        Returns:
            dict: Training history
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                model_path = self.exp_dir / 'best_model.pth'
                save_model(
                    self.model,
                    model_path,
                    metadata={
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'config': vars(self.config)
                    }
                )
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            
            print("-" * 60)
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        # Save training history
        history_path = self.exp_dir / 'training_history.json'
        save_results(self.history, history_path)
        
        # Test the best model
        print("\nTesting best model...")
        best_model_path = self.exp_dir / 'best_model.pth'
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = self.test()
        
        # Save final results
        final_results = {
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'training_time_minutes': training_time / 60,
            'config': vars(self.config)
        }
        
        results_path = self.exp_dir / 'final_results.json'
        save_results(final_results, results_path)
        
        return self.history


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    experiment_name=None
):
    """
    Convenience function to train a model
    
    Args:
        model: CNN model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration object
        experiment_name: Optional experiment name
        
    Returns:
        tuple: (trained model, training history)
    """
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Create trainer
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=experiment_name
    )
    
    # Train
    history = trainer.train()
    
    return trainer.model, history


if __name__ == "__main__":
    # This will be used for testing later
    print("CNN Training Module")
    print("Import this module to train models")
