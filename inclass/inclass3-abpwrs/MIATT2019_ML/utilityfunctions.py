import itk
import numpy as np

def display_overlay_callback(t1_image, t2_image, label, z, opacity=0.0):
    """Display label map overlay on image
    
    Args:
      t1_image (itk.Image): Input t1 image
      t2_image (itk.Image): Input t2 image
      label (itk.Image): Label map image
      z (int): Image slice to display
      opacity (float): Opacity of overlay
     """
    img_slices = [t1_image[:,:,z], t2_image[:,:,z]]
    lbl_slices = [label[:,:,z], label[:,:,z]]
    img = sitk.Tile(img_slices, [2,1])
    lbl = sitk.Tile(lbl_slices, [2,1])
    overlay = sitk.LabelOverlay(img, lbl, opacity)
    myshow(overlay, dpi=30)
    


    img_t1_255 = itk.Cast(sitk.RescaleIntensity(t1_img), sitk.sitkUInt8)
    img_t2_255 = itk.Cast(sitk.RescaleIntensity(t2_img), sitk.sitkUInt8)
    interact(display_overlay_callback,
         t1_image = fixed(img_t1_255),
         t2_image = fixed(img_t2_255),
         label = fixed(lbl_img), 
         z=(0,t1_img.GetSize()[2]),
         opacity=(0.0,1.0)
        )



# Utility: run this

def flatten_image(input_image):
    """Flattens image into a 1D array.
    
    Args:
      input_image (itk.Image): Input image
    Returns:
      np_image_as_array (np.array): 1D array representing a image
      
    """
    feat_img = itk.GetArrayFromImage(input_image)
    np_image_as_array = feat_img.flatten()  
   
    return np_image_as_array


def get_class_array(is_class, lbl_image, image):
    """Extracts image data with label map value of is_class
    
    Args:
      is_class (str): class code
      lbl_image (itk.Image): Label map
      image (itk.Image): Input image
    Returns:
      t (np.array): Image data corresponding to is_class label
    
    """
        
    lbl_flat = flatten_image(lbl_image)
    img_flat = flatten_image(image)
    temp0 = np.array([])
    temp1 = img_flat[lbl_flat == is_class]
    t = np.hstack((temp0, temp1))
    return t