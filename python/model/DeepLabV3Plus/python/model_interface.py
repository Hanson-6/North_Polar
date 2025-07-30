"""
Model Interface for DeepLabV3Plus
=================================

This module provides an interface between the GMap Area AI tool 
and the DeepLabV3Plus model for building segmentation.

Author: Generated for integration
Version: 1.0.0
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import logging

# Setup logging
logger = logging.getLogger('ModelInterface')

# Add model path to system path
model_path = os.path.join(os.getcwd(), 'model')
if model_path not in sys.path:
    sys.path.insert(0, model_path)

# Try to import the model
# try:
from inference import DeepLabV3PlusInference
MODEL_AVAILABLE = True
# except ImportError:
#     logger.warning("DeepLabV3Plus model not found. Using mock predictions.")
#     MODEL_AVAILABLE = False


class DeepLabV3PlusInterface:
    """
    Interface for DeepLabV3Plus model inference.
    
    This class provides a simple API for the GMap tool to use
    the DeepLabV3Plus model for building segmentation.
    """
    
    def __init__(self, checkpoint_path=None):
        """
        Initialize the model interface.
        
        Args:
            checkpoint_path: Path to model checkpoint. If None, will search for latest.
        """
        self.model = DeepLabV3PlusInference(
            checkpoint_path=checkpoint_path,
            num_classes=2,  # Change if using different classes
            device=None,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
            
        # Load model if available
        if MODEL_AVAILABLE and checkpoint_path:
            # try:
            self.model = DeepLabV3PlusInference(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
                num_classes=2
            )
            logger.info(f"Model loaded successfully from {checkpoint_path}")
            # except Exception as e:
            #     logger.error(f"Failed to load model: {e}")
            #     self.model = None
        else:
            logger.warning("Model not available, will use mock predictions")
            
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoints directory."""
        checkpoint_dir = os.path.join(os.getcwd(), 'model', 'result', 'checkpoints')
        
        # Alternative paths to check
        alternative_paths = [
            os.path.join(os.getcwd(), 'result', 'checkpoints'),
            os.path.join(os.getcwd(), 'checkpoints'),
            os.path.join(os.getcwd(), 'model', 'checkpoints'),
        ]
        
        # Check all possible paths
        for path in [checkpoint_dir] + alternative_paths:
            if os.path.exists(path):
                checkpoint_dir = path
                break
        else:
            logger.warning(f"Checkpoint directory not found")
            return None
            
        # Find all .pth files
        # try:
        pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not pth_files:
            logger.warning(f"No checkpoint files found in {checkpoint_dir}")
            return None
            
        # Sort by modification time and get the latest
        pth_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
        latest_checkpoint = os.path.join(checkpoint_dir, pth_files[0])
        
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
            
        # except Exception as e:
        #     logger.error(f"Error finding checkpoint: {e}")
        #     return None
            
    def predict_area(self, image_data):
        """
        Predict building segmentation for a given area.
        
        Args:
            image_data: Dictionary containing:
                - 'image': PIL Image object
                - 'image_array': numpy array of the image
                - 'bounds': boundary information
                - 'geometry': geometry information
                - 'size': image size tuple
                
        Returns:
            numpy.ndarray: Binary mask where 1 indicates buildings, 0 indicates background
        """
        # try:
        # Extract image
        if isinstance(image_data, dict):
            if 'image' in image_data and isinstance(image_data['image'], Image.Image):
                image = image_data['image']
            elif 'image_array' in image_data:
                image = Image.fromarray(image_data['image_array'])
            else:
                raise ValueError("No valid image data found in input")
        else:
            raise ValueError("Expected dictionary input with image data")
            
        # Get image size
        target_size = image_data.get('size', (256, 256))
        
        # If model is available, use it
        if self.model is not None:
            logger.info("Running model inference...")
            mask = self.model.predict(image, target_size=target_size)
            logger.info(f"Inference complete. Mask shape: {mask.shape}")
            return mask
        else:
            # Create mock prediction for testing
            logger.warning("Using mock prediction (model not available)")
            return self._create_mock_prediction(image, target_size)
                
        # except Exception as e:
        #     logger.error(f"Error during prediction: {e}")
        #     # Return empty mask on error
        #     return np.zeros(target_size, dtype=np.uint8)
            
    def _create_mock_prediction(self, image, target_size):
        """
        Create a mock prediction for testing when model is not available.
        
        This creates a simple threshold-based "prediction" that can be used
        for testing the UI workflow.
        """
        # Resize image to target size
        if image.size != target_size:
            image = image.resize(target_size)
            
        # Convert to grayscale and threshold
        gray = image.convert('L')
        gray_array = np.array(gray)
        
        # Simple threshold-based mock segmentation
        # This is just for testing - real model would be much better
        threshold = 128
        mask = (gray_array > threshold).astype(np.uint8)
        
        # Add some morphological operations to make it look more realistic
        from scipy import ndimage
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)
        
        return mask.astype(np.uint8)
        
    def predict_batch(self, image_list, target_size=(256, 256)):
        """
        Predict building segmentation for multiple areas.
        
        Args:
            image_list: List of image data dictionaries
            target_size: Target size for all images
            
        Returns:
            List of numpy arrays (masks)
        """
        masks = []
        for image_data in image_list:
            mask = self.predict_area(image_data)
            masks.append(mask)
        return masks
        
    def get_model_info(self):
        """Get information about the loaded model."""
        if self.model is not None:
            return {
                'available': True,
                'device': str(self.device),
                'num_classes': 2,
                'input_size': (256, 256)
            }
        else:
            return {
                'available': False,
                'reason': 'Model not loaded',
                'device': str(self.device)
            }


# Utility functions for post-processing

def smooth_mask(mask, kernel_size=5):
    """
    Smooth a binary mask using morphological operations.
    
    Args:
        mask: Binary mask (numpy array)
        kernel_size: Size of the smoothing kernel
        
    Returns:
        Smoothed mask
    """
    from scipy import ndimage
    
    # Apply morphological closing followed by opening
    mask = ndimage.binary_closing(mask, iterations=kernel_size//2)
    mask = ndimage.binary_opening(mask, iterations=kernel_size//2)
    
    return mask.astype(np.uint8)


def mask_to_geojson(mask, bounds):
    """
    Convert a binary mask to GeoJSON format.
    
    Args:
        mask: Binary mask (numpy array)
        bounds: Geographic bounds of the mask
        
    Returns:
        GeoJSON dictionary
    """
    # This is a placeholder - actual implementation would need
    # proper georeferencing and polygon extraction
    return {
        "type": "FeatureCollection",
        "features": []
    }


def overlay_mask_on_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay a mask on an image with transparency.
    
    Args:
        image: PIL Image or numpy array
        mask: Binary mask
        alpha: Transparency level (0-1)
        color: RGB color tuple for mask
        
    Returns:
        PIL Image with overlay
    """
    # Convert inputs to PIL Images
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        
    # Create colored overlay
    overlay = Image.new('RGB', image.size, color)
    
    # Apply mask as alpha channel
    image = image.convert('RGBA')
    overlay = overlay.convert('RGBA')
    mask = mask.convert('L')
    
    # Composite images
    overlay.putalpha(mask)
    result = Image.alpha_composite(image, overlay)
    
    return result.convert('RGB')


# Test function
if __name__ == "__main__":
    # Test the interface
    interface = DeepLabV3PlusInterface()
    print("Model info:", interface.get_model_info())
    
    # Create test image
    test_image = Image.new('RGB', (256, 256), color='white')
    test_data = {
        'image': test_image,
        'size': (256, 256),
        'bounds': None,
        'geometry': None
    }
    
    # Test prediction
    mask = interface.predict_area(test_data)
    print(f"Prediction shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")