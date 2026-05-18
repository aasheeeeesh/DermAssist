import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import visualization as viz
from .config import paths, preprocessing_config

def denormalize_image(image_tensor):
    """
    Reverses the PyTorch normalization applied during preprocessing 
    so the image can be displayed normally on screen.
    """
    mean = np.array(preprocessing_config.NORMALIZE_MEAN).reshape(3, 1, 1)
    std = np.array(preprocessing_config.NORMALIZE_STD).reshape(3, 1, 1)
    
    # Reverse the math: (pixel * std) + mean
    denorm_image = image_tensor.numpy() * std + mean
    
    # Clip to valid range [0, 1] and rearrange dimensions for matplotlib (H, W, C)
    denorm_image = np.clip(denorm_image, 0, 1)
    return np.transpose(denorm_image, (1, 2, 0))

def plot_xai_heatmap(image_tensor, attribution_tensor, save_name="xai_heatmap.png"):
    """
    Uses Captum's built-in visualization tools to overlay the 
    importance heatmap on top of the original skin lesion image.
    """
    # 1. Denormalize original image for viewing
    original_img_np = denormalize_image(image_tensor)
    
    # 2. Convert attribution tensor for viewing (H, W, C)
    attr_np = np.transpose(attribution_tensor.numpy(), (1, 2, 0))
    
    # 3. Create a side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axes[0].imshow(original_img_np)
    axes[0].axis('off')
    axes[0].set_title("Original Skin Lesion")
    
    # Plot heatmap overlay using Captum's visualizer
    viz.visualize_image_attr(
        attr_np,
        original_img_np,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title="AI Attention Heatmap",
        plt_fig_axis=(fig, axes[1]),
        use_pyplot=False
    )
    
    # Save the figure
    save_path = paths.REPORTS_DIR / "figures" / save_name
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved XAI Visualization to {save_path}")

def plot_pca_scatter(X_pca, y, class_names=None, filename="pca_scatter.png"):
    """
    Creates a 2D scatter plot of the PCA or FDA embeddings.
    """
    plt.figure(figsize=(10, 8))
    
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for idx, c in enumerate(classes):
        label_name = class_names[c] if class_names else f"Class {c}"
        subset = X_pca[y == c]
        plt.scatter(subset[:, 0], subset[:, 1], 
                    color=colors[idx], label=label_name, alpha=0.6, edgecolors='w')
        
    plt.title("2D Projection of CNN Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = paths.REPORTS_DIR / "figures" / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved 2D Scatter Plot to {save_path}")
