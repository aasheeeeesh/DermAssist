import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from tqdm import tqdm
from .config import model_config, paths

def get_feature_extractor(model_name="resnet50"):
    """
    Loads a pretrained CNN and removes its final classification layer 
    so it outputs raw feature embeddings instead of class probabilities.
    """
    print(f"Loading pretrained {model_name}...")
    
    if model_name == "resnet50":
        # Load the standard ResNet50 trained on ImageNet
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        
        # ResNet's last layer is 'fc' (fully connected). We replace it with an Identity layer
        # This means the network stops right before the final classification, giving us a 2048-dim vector.
        model.fc = nn.Identity()
        embedding_dim = 2048
        
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet's classifier is a Sequential block. We replace it.
        model.classifier = nn.Identity()
        embedding_dim = 1280
        
    else:
        raise ValueError(f"Model {model_name} not supported yet.")

    # Move model to GPU if available and set to Evaluation Mode
    device = torch.device(model_config.DEVICE)
    model = model.to(device)
    model.eval()  # Very important: turns off Dropout and BatchNorm updates
    
    return model, embedding_dim

def extract_embeddings(dataloader, model):
    """
    Passes all images in the dataloader through the model to extract embeddings.
    """
    device = torch.device(model_config.DEVICE)
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings...")
    # torch.no_grad() disables gradient calculation, saving huge amounts of memory and time
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Batches"):
            images = images.to(device)
            
            # Forward pass: images go in, feature vectors come out
            features = model(images)
            
            # Move features back to CPU and convert to numpy array
            all_embeddings.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate all batches into one giant numpy matrix
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    return final_embeddings, final_labels

def save_embeddings(embeddings, labels, split_name):
    """
    Saves the extracted feature vectors to the disk.
    This exactly mimics the .npz format you used in your SML assignments!
    """
    save_path = paths.EMBEDDINGS_DIR / f"{split_name}_embeddings.npz"
    print(f"Saving {split_name} embeddings to {save_path}...")
    
    # Save as compressed numpy file
    np.savez_compressed(
        save_path, 
        embeddings=embeddings, 
        labels=labels
    )
    print(f"Saved {embeddings.shape[0]} samples with shape {embeddings.shape[1]}")

def load_embeddings(split_name):
    """
    Loads previously saved embeddings from the disk.
    """
    load_path = paths.EMBEDDINGS_DIR / f"{split_name}_embeddings.npz"
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        
    data = np.load(load_path)
    return data['embeddings'], data['labels']
