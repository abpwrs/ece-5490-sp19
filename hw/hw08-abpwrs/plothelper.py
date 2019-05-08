import itk
import numpy as np

from erodehelper import ErodeLabels
from medianhelper import MedianFilter
from utilityfunctions import get_class_array

from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import gridspec

class PlotClassDistribution:
    """Plots joint histogram of two images.
    
    The joint histogram of two images is plotted for each class label separately.
    The image can be smoothed, or the class labels can be removed using
    erosion and the joint histogram will update.
    
    Attributes:
        median_T1 : instance of a MedianFilter
        median_T2 : instance of a MedianFilter
        erode_instancec : instance of a ErodeLabels
        label_map (itk.Image): Label map for images 
        image1 (itk.image): Input image
        image2 (itk.image): Input image
    
    """
    def __init__(self, img1, img2, img_label, lbldict):
        """Initializes attributes
    
        Args:
            img1 (itk.Image): Input image
            img2 (itk.Image): Input image
            img_label (itk.Image): Label map image
        """
        self.image1 = img1
        self.image2 = img2
        self.label_map = img_label
        self.LABELS=lbldict
        self.median_T1 = MedianFilter(self.image1)
        self.median_T2 = MedianFilter(self.image2)
        self.erode_instance = ErodeLabels(img_label)
        
    def plot_histogram(self, x, y, title=None):
        """Plots the joint histogram of two images
    
        http://www.bi.wisc.edu/~fox//2013/06/05/visualizing-the-correlation-of-two-volumes/
    
        Args:
            x (np.array): Array represending an image
            y (np.array): Array represending an image
            title (str): Title for figure. 
 
        """
        #mainFig = plt.figure(1, figsize=(5, 5), facecolor='white')
        mainFig = plt.figure(1, facecolor='white')
        # define some gridding.
        axHist2d = plt.subplot2grid((9, 9), (1, 0), colspan=8, rowspan=8)
        axHistx = plt.subplot2grid((9, 9), (0, 0), colspan=8)
        axHisty = plt.subplot2grid((9, 9), (1, 8), rowspan=8)

        # show joint histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=(100, 100), range=[[200, 3800], [200, 3800]])
        axHist2d.imshow(H.T, interpolation='nearest', aspect='auto', cmap='jet')

        # show individual histograms
        axHistx.hist(x, bins=xedges, facecolor='blue', alpha=0.5, edgecolor='None')
        axHisty.hist(y, bins=yedges, facecolor='blue', alpha=0.5, orientation='horizontal', edgecolor='None')

        # formatting plot
        axHistx.set_xlim([xedges.min(), xedges.max()])
        axHisty.set_ylim([yedges.min(), yedges.max()])
        axHist2d.set_ylim([axHist2d.get_ylim()[1], axHist2d.get_ylim()[0]])
        
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHistx.yaxis.set_major_formatter(nullfmt)
        axHisty.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)
        axHistx.spines['top'].set_visible(False)
        axHistx.spines['right'].set_visible(False)
        axHistx.spines['left'].set_visible(False)
        axHisty.spines['top'].set_visible(False)
        axHisty.spines['bottom'].set_visible(False)
        axHisty.spines['right'].set_visible(False)
        axHistx.set_xticks([])
        axHistx.set_yticks([])
        axHisty.set_xticks([])
        axHisty.set_yticks([])
        myTicks = np.arange(0, 100, 10)
        axHist2d.set_xticks(myTicks)
        axHist2d.set_yticks(myTicks)
        axHist2d.set_xticklabels(np.round(xedges[myTicks], 2), rotation=45)
        axHist2d.set_yticklabels(np.round(yedges[myTicks], 2))

        # add labels
        axHist2d.set_xlabel('T1', fontsize=16)
        axHist2d.set_ylabel('T2', fontsize=16)
        axHistx.set_title('T1', fontsize=10)
        axHisty.yaxis.set_label_position("right")
        axHisty.set_ylabel('T2', fontsize=10, rotation=-90, verticalalignment='top', horizontalalignment='center')

        mainFig.canvas.set_window_title(('T1 vs T2'))
        if title:
            plt.title(title, loc='left')
        plt.show()

    def display(self, erode_radius=0, median_radius=0):
        """Displays joint histogram
        
        Args:
            erode_radius (int): radius of structuring element
            median_radius (int): radius of filter
            
        """
        

        self.median_T1.set_radius(median_radius)
        self.median_T2.set_radius(median_radius)
        t1_image = self.median_T1.run()
        t2_image = self.median_T2.run()
        
        self.erode_instance.set_radius((erode_radius, erode_radius, erode_radius))
        lbl_image = self.erode_instance.run()
        #rgb_img = itk.label_to_rgb_image_filter(Input=lbl_image)
        #slices = [t1_image[:, :, t1_image.GetSize()[2]//2], t2_image[:, :, t2_image.GetSize()[2]//2]]
        #tile = sitk.Tile(slices, [2, 1])
        #myshow(tile, dpi=50)
        
        t1 = get_class_array(self.LABELS['WM'], lbl_image, t1_image)
        t2 = get_class_array(self.LABELS['WM'], lbl_image, t2_image)
        plt.figure(figsize=(4,4))
        self.plot_histogram(t1, t2, "White Matter") 
        
        t1 = get_class_array(self.LABELS['GM'], lbl_image, t1_image)
        t2 = get_class_array(self.LABELS['GM'], lbl_image, t2_image)
        plt.figure(figsize=(4,4))
        self.plot_histogram(t1, t2, "Gray Matter") 
        
        t1 = get_class_array(self.LABELS['CSF'], lbl_image, t1_image)
        t2 = get_class_array(self.LABELS['CSF'], lbl_image, t2_image)
        plt.figure(figsize=(4,4))
        self.plot_histogram(t1, t2, "CSF") 