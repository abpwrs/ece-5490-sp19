"""
A helper class for wrapping erode labels image filtering
"""
import itk
from itkwidgets import view
import itkwidgets

from ipywidgets import interact, fixed
from ipywidgets import interactive
import ipywidgets as widgets

class ErodeLabels:
    """Erodes each label of a label map. 
    
    The ErodeLabels class uses itk.BinaryErodeImageFilter to
    erode a label map. All labels are eroded independently.
    Need to set the radius for the erosion structuring element.
    All errosions are stored in a cache for time efficiency.
    
    Attributes:
        erode_filter : instance of an itk.BinaryErodeImageFilter
        cached (dict): stores filter output for efficency
        radius (int): radius of median filter 
        labels (list): Labels to be eroded
        image (itk.Image): Input image to be filtered
    
    """
    
    def __init__(self, img):
        """Initializes attributes
    
        Args:
            image (sitk.Image): Input label map to be eroded 
        """

        self.radius = (0,0,0)
        self.labels = [1, 2, 4]
        self.image = img
        self.cached = { self.radius: self.image }
        self.viewer = view( self.image,
              ui_collapsed=True, annotations=True, interpolation=True,  cmap=itkwidgets.cm.GnYlRd,
              mode='x', shadow=True, slicing_planes=False, gradient_opacity=0.22)
        self.min_radius=0
        self.max_radius=4
        self.slider = interactive( self.display, 
                                     WMradius=(self.min_radius, self.max_radius), 
                                     GMradius=(self.min_radius, self.max_radius), 
                                     CSFradius=(self.min_radius, self.max_radius)
                                 )
     
    def set_radius(self, radius=(0,0,0)):
        """Set the radius of the erode structure element
    
        Args:
            radius (int): Radius of structuring element 
        """
        self.radius = radius

    def run(self):
        """Performs the erosion"""
        erode_tuple = self.radius
        if(erode_tuple not in self.cached):
            erode_img = self.image
            for ind in range(len(self.labels)):
                erode_img = itk.binary_erode_image_filter( Input=erode_img, Radius=self.radius[ind], ForegroundValue= self.labels[ind])
            self.cached[erode_tuple] = erode_img
        return self.cached[erode_tuple]

    def display(self, WMradius = 0, GMradius = 0, CSFradius = 0):
        """Displays eroded label map
        Args:
            label: label key
            label_dict (dict): maps label key to label values to erode
            radius (int): radius of structuring element
        """
        self.radius = (WMradius, GMradius, CSFradius)
        self.run()
        #        rgb_img = itk.label_to_rgb_image_filter(Input=self.cached[self.radius])
        print('radius = {}'.format(self.radius))
        self.viewer.image=self.cached[self.radius]