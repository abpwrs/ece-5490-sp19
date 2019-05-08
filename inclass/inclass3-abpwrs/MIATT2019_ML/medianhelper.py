"""
A helper class for wrapping median image filtering
"""
import itk
from itkwidgets import view

from ipywidgets import interact, fixed
from ipywidgets import interactive
import ipywidgets as widgets

class MedianFilter:
    """Performs median filtering on an image.
    
    The MedianFilter class uses itk.MedianImageFilter to
    smooth an image. The radius of the median filter is 1 by
    default, can set using member function set_radius(radius).
    All outputs are stored in a cache for time efficiency.
    
    Attributes:
        median_filter : instance of an itk.MedianImageFilter
        cached (dict): stores filter output for efficency
        radius (int): radius of median filter 
        image (itk.image): Input image to be filtered
    """
    
    def __init__(self, img):
        """Initializes attributes
    
        Args:
            image (itk.Image): Input image to be filtered 
 
        """
        self.image = img
        self.cached = {0: img} # initialize dictionary with zero filtering
        self.radius = 1
        self.min_radius = 0
        self.max_radius = 3
        self.slider_step_size = 1
        self.viewer = view( self.image,
              ui_collapsed=True, annotations=True, interpolation=True,  cmap='Grayscale',  
              mode='x', shadow=True, slicing_planes=False, gradient_opacity=0.22)
        self.slider = interactive( self.display, radius=(self.min_radius, self.max_radius, self.slider_step_size) )
        
        ## Pre-cache results
        for i in range(self.min_radius,self.max_radius,self.slider_step_size):
            import time
            start = time.time()
            self.set_radius(i)
            self.run()
            end = time.time()
            print("Median filter with size {0} took {1} seconds to evaluate".format(i, end - start))

    def set_radius(self, radius):
        """Set the radius of the filter
    
        Args:
            radius (int): Radius of filter 
 
        """
        self.radius = radius

    def run(self):
        """Performs median smoothing"""
        if self.radius not in self.cached:
            self.cached[self.radius] = itk.median_image_filter(Input=self.image,Radius=self.radius)
        return self.cached[self.radius]

    def display(self, radius=0):
        """Displays filtered images
        
        Args:
            radius (int): radius of filter
        """
        self.set_radius(radius)
        self.run()
        self.viewer.image = self.cached[self.radius]
        print(self.radius)
