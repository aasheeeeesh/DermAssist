import torch
import numpy as np
from captum.attr import Occlusion
from .config import preprocessing_config

class HybridPipelineWrapper:
    """
    Captum XAI requires a PyTorch function to analyze. Because our pipeline uses 
    a PyTorch CNN followed by a Scikit-Learn classifier, we wrap them together 
    into a single "black box" function.
    """
    def __init__(self, feature_extractor, classifier, device):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.device = device
        
    def __call__(self, input_image):
        """
        Takes an image tensor, extracts features, and returns the classical model's probabilities.
        """
        # 1. Pass image through CNN to get 2048-dim embeddings
        with torch.no_grad():
            features = self.feature_extractor(input_image).cpu().numpy()
            
        # 2. Pass embeddings through the Scikit-Learn classifier
        probs = self.classifier.predict_proba(features)
        
        # 3. Return probabilities as a PyTorch tensor (required by Captum)
        return torch.tensor(probs, dtype=torch.float32).to(self.device)

def generate_occlusion_heatmap(image_tensor, feature_extractor, classifier, target_class=1):
    """
    Generates an Explainable AI (XAI) heatmap using the Occlusion method.
    
    How it works: It slides a gray square across the image. If covering a specific 
    spot causes the SVM's probability of "Cancer" to drop drastically, that spot 
    is marked as highly important (red). If covering a spot does nothing, it's marked 
    as unimportant (blue/transparent).
    """
    device = next(feature_extractor.parameters()).device
    
    # Create the unified pipeline wrapper
    wrapper = HybridPipelineWrapper(feature_extractor, classifier, device)
    
    # Initialize the Occlusion algorithm
    occlusion = Occlusion(wrapper)
    
    # Define the sliding window size. 
    # For a 224x224 image, a 20x20 pixel patch is a good size for detecting lesion features.
    # The '3' corresponds to the RGB color channels.
    window_size = (3, 20, 20) 
    stride = (3, 10, 10)
    
    # Ensure image has a batch dimension: shape becomes [1, 3, 224, 224]
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    image_tensor = image_tensor.to(device)
    
    print(f"Generating XAI heatmap for target class {target_class}...")
    
    # Generate the attribution map
    attributions = occlusion.attribute(
        image_tensor,
        strides=stride,
        target=target_class,
        sliding_window_shapes=window_size,
        show_progress=False
    )
    
    # Remove batch dimension and return to CPU for plotting
    return attributions.squeeze(0).cpu()
