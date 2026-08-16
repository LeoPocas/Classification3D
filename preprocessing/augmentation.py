import numpy as np
import scipy.ndimage as ndimage

def random_rotation_3d(volume, max_angle=25):
    """
    Applies a random rotation to the 3D volume in the XY plane (axial).
    
    Args:
        volume: 3D numpy array (H, W, D) or (H, W, D, C)
        max_angle: Maximum angle in degrees for rotation (default: 25)
    
    Returns:
        Rotated 3D volume.
    """
    # Random angle between -max_angle and +max_angle
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Rotate in the XY plane (axes 0 and 1). 
    # reshape=False keeps the original shape (cropping corners if necessary)
    # order=1 (linear interpolation) is faster and usually sufficient
    if volume.ndim == 4:
        # If channel dimension exists, rotate each channel (though usually 1)
        # Assuming (H, W, D, C), we rotate H,W.
        volume_aug = np.zeros_like(volume)
        for c in range(volume.shape[-1]):
            volume_aug[..., c] = ndimage.rotate(volume[..., c], angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)
        return volume_aug
    else:
        return ndimage.rotate(volume, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0.0)

def random_zoom_3d(volume, zoom_range=(0.9, 1.1)):
    """
    Applies a random zoom (scaling) to the 3D volume.
    
    Args:
        volume: 3D numpy array
        zoom_range: Tuple (min_zoom, max_zoom)
        
    Returns:
        Zoomed 3D volume (cropped or padded to match original shape).
    """
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    
    # We apply zoom to H and W, usually keeping D (depth/slices) intact or scaling it too?
    # Medical volumes often have anisotropic resolution. Scaling Z might misalign with slice thickness.
    # Let's scale H and W mainly. If we scale Z, we change the physical length of the heart covered.
    # Let's scale all 3 dimensions uniformly to preserve aspect ratio of features, 
    # but we must handle the shape change.
    
    if volume.ndim == 4:
        h, w, d, c = volume.shape
        # Zoom factor array: (H_scale, W_scale, D_scale, C_scale)
        # We don't scale the channel dimension
        factors = (zoom_factor, zoom_factor, zoom_factor, 1)
        volume_zoomed = ndimage.zoom(volume, factors, order=1, mode='constant', cval=0.0)
    else:
        h, w, d = volume.shape
        factors = (zoom_factor, zoom_factor, zoom_factor)
        volume_zoomed = ndimage.zoom(volume, factors, order=1, mode='constant', cval=0.0)
        
    # Crop or Pad to restore original shape
    new_shape = volume_zoomed.shape
    orig_shape = volume.shape
    
    # Result container
    final_vol = np.zeros(orig_shape, dtype=volume.dtype)
    
    # Calculate crop/pad coordinates
    # For each dimension (except channel)
    slices_src = []
    slices_dest = []
    
    for i in range(min(3, len(orig_shape))): # Iterate H, W, D
        orig_len = orig_shape[i]
        new_len = new_shape[i]
        
        if new_len > orig_len:
            # Crop center
            start = (new_len - orig_len) // 2
            slices_src.append(slice(start, start + orig_len))
            slices_dest.append(slice(0, orig_len))
        else:
            # Pad center
            start = (orig_len - new_len) // 2
            slices_src.append(slice(0, new_len))
            slices_dest.append(slice(start, start + new_len))
            
    if volume.ndim == 4:
        slices_src.append(slice(None)) # All channels
        slices_dest.append(slice(None))
        
    final_vol[tuple(slices_dest)] = volume_zoomed[tuple(slices_src)]
    
    return final_vol

def apply_augmentation(volume, config_str):
    """
    Applies data augmentation based on the config string.
    Supports combinations separated by '+' (e.g., 'rotate+zoom').
    
    Keywords: 'rotate', 'zoom', 'none'
    """
    if not config_str or str(config_str).lower() == 'none' or str(config_str) == 'False':
        return volume

    methods = str(config_str).lower().split('+')
    
    # Apply methods in sequence
    augmented_volume = volume.copy()
    
    if 'rotate' in methods:
        augmented_volume = random_rotation_3d(augmented_volume)
        
    if 'zoom' in methods:
        augmented_volume = random_zoom_3d(augmented_volume)
        
    return augmented_volume
