import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import paths, preprocessing_config

# Set a random seed for reproducibility (following the style from your SML assignments)
np.random.seed(42)

class SkinLesionDataset(Dataset):
    """
    A custom PyTorch Dataset for loading skin lesion images.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list or numpy array): Paths to the image files.
            labels (list or numpy array): Corresponding integer labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        # Convert image to RGB
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(is_train=True):
    """
    Returns the torchvision transforms for preprocessing images.
    Instead of manually dividing by 255.0 as done in SML assignments, 
    ToTensor() handles the 0-1 scaling, and Normalize handles the centering.
    """
    if is_train:
        # Include data augmentation for the training set
        return transforms.Compose([
            transforms.Resize(preprocessing_config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(), # This automatically divides by 255.0
            transforms.Normalize(
                mean=preprocessing_config.NORMALIZE_MEAN,
                std=preprocessing_config.NORMALIZE_STD
            )
        ])
    else:
        # Only resize and normalize for validation/test sets
        return transforms.Compose([
            transforms.Resize(preprocessing_config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing_config.NORMALIZE_MEAN,
                std=preprocessing_config.NORMALIZE_STD
            )
        ])

def create_manual_splits(image_paths, labels, val_ratio=0.15, test_ratio=0.15):
    """
    Manually creates train/val/test splits using randomized index shuffling.
    This logic mimics the explicit manual splitting used in your SML A4 assignment.
    """
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    # Randomly shuffle indices to ensure unbiased validation/test sets
    np.random.shuffle(indices)
    
    val_size = int(n_samples * val_ratio)
    test_size = int(n_samples * test_ratio)
    
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    return train_indices, val_indices, test_indices

def get_dataloaders(csv_file_path, image_dir):
    """
    Main function to orchestrate the loading and preprocessing pipeline.
    Expects a CSV with 'image_id' and 'label' columns.
    """
    if not os.path.exists(csv_file_path):
        print(f"Warning: Metadata file {csv_file_path} not found. Returning empty loaders.")
        return None, None, None
        
    df = pd.read_csv(csv_file_path)
    
    # Construct full image paths
    # Assuming images are stored as .jpg
    image_paths = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in df['image_id']]
    labels = df['label'].values
    
    # Filter classes if needed (similar to train_mask = (y_train == 4) | (y_train == 9) in A4)
    # For DermAssist, we'll assume the CSV is already pre-filtered.
    
    print(f"Total samples found: {len(labels)}")
    
    # Create splits
    train_idx, val_idx, test_idx = create_manual_splits(image_paths, labels)
    
    image_paths = np.array(image_paths)
    
    # Create Datasets
    train_dataset = SkinLesionDataset(
        image_paths[train_idx], 
        labels[train_idx], 
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = SkinLesionDataset(
        image_paths[val_idx], 
        labels[val_idx], 
        transform=get_transforms(is_train=False)
    )
    
    test_dataset = SkinLesionDataset(
        image_paths[test_idx], 
        labels[test_idx], 
        transform=get_transforms(is_train=False)
    )
    
    print(f"Train set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=preprocessing_config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=preprocessing_config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=preprocessing_config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=preprocessing_config.NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=preprocessing_config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=preprocessing_config.NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the logic
    print("Preprocessing module ready.")
