# This file must be self contained
# such that the functions/classes variables in
# this file can be independantly imported
# for validation

import os
import SimpleITK as sitk
import itk
import glob
from enum import Enum
from math import floor
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy.ndimage import generic_filter
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# Where the data lives
GLB_DATA_DIR = '/nfsscratch/opt/ece5490/data/MIATT_EYES'

# How the classes relate to the label maps
CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ['bg', 'gm', 'wm', 'csf']
INPUT_SHAPE = [None, 128, 128, 128, 1]

'''
| Image Range   | Description     |
| ------------- | -------------   |
| 0001 - 0100   | Hidden Tests    |
| 0101 - 0200   | Test Data       |
| 0201 - 0300   | Validation Data |
| 0301 - 0804   | Trainging Data  |
'''


# class for the mode
class ModeKeys(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    HIDDEN_TEST = 3


# This class kind of exploded and should be split into sub classes
class HCPData(object):
    """
    HCPData is a class that is intended to encapsulate
    all the necessary information related to
    * managing groups of subjects
    * retrieving features
    * batch-wise processing
    * retrieving labels
    """

    def __init__(self, feature_type='image_wise'):
        # image size
        # self.image_size = [200, 200, 200]
        self.cleaner = Cleaner()

        # feature extraction type
        self.ALLOWED_FEATURE_TYPES = ['voxel_wise', 'image_wise']
        if feature_type not in self.ALLOWED_FEATURE_TYPES:
            raise Exception("invalid feature type")
        self.feature_type = feature_type

        # extract subject ids
        self.subjects = glob.glob(os.path.join(GLB_DATA_DIR, '*'))
        self.subjects.sort()
        self.subjects = list(map(lambda x: Subject(os.path.basename(x)), self.subjects))

        # this could be voxels or whole image based on
        self.batch_size = 50
        self.batch_index = None
        self.batch_subs = None

        # encoder
        self.encoder = None

        # set bounds for image ranges
        self.MIN_D = {
            ModeKeys.TRAIN: 300,
            ModeKeys.VAL: 200,
            ModeKeys.TEST: 100,
            ModeKeys.HIDDEN_TEST: 0,
        }
        self.MAX_D = {
            ModeKeys.TRAIN: 804,
            ModeKeys.VAL: 299,
            ModeKeys.TEST: 199,
            ModeKeys.HIDDEN_TEST: 99,
        }

    def get_subjects(self, mode=None):
        if mode:
            sub_list = list(
                filter(
                    lambda x: self.MIN_D[mode] <= int(x.get_subject_id()) <= self.MAX_D[mode], self.subjects
                )
            )
            return sub_list
        # if no mode is given
        return self.subjects

    def init_batches(self, mode):
        self.batch_index = 0
        self.batch_subs = self.get_subjects(mode=mode)

    def next_batch(self):
        curr_subs = self.batch_subs[self.batch_index:(self.batch_index + self.batch_size)]
        self.batch_index += self.batch_size

        print("Loading & cleaning batch, this may be a moment...")
        images = []
        labels = []
        if self.feature_type == 'image_wise':
            for sub in curr_subs:
                print("Loading {}".format(str(sub)))
                try:
                    im = np.expand_dims(sitk.GetArrayFromImage(
                        sitk.Cast(
                            self.cleaner.clean(image=sub.get_t1_image(), is_label_map=False), sitk.sitkUInt8
                        )
                    ), 3)
                    images.append(im)

                    lbl = np.expand_dims(sitk.GetArrayFromImage(
                        sitk.Cast(
                            self.cleaner.resize(image=sub.get_label_mask(), is_label_map=True), sitk.sitkUInt8
                        )
                    ), 3)
                    labels.append(lbl)
                except Exception as e:
                    print("Subject {} Failed".format(sub.get_subject_id()))
        else:
            raise NotImplementedError
            # do voxel-wise feature extraction -- need to sub-sample the image somehow
            # for sub in curr_subs:
            #     print("Loading {}".format(str(sub)))
            #     try:
            #         im = sitk.GetArrayFromImage(
            #             sitk.Cast(
            #                 self.cleaner.clean(image=sub.get_t1_image(), is_label_map=False), sitk.sitkUInt8
            #             )
            #         )
            #         prob_gm = sub.get_gm_prob_mask()
            #         prob_wm = sub.get_wm_prob_mask()
            #         prom_csf = sub.get_csf_prob_mask()
            #
            #         lbl = sitk.GetArrayFromImage(
            #             sitk.Cast(
            #                 self.cleaner.resize(image=sub.get_label_mask(), is_label_map=True), sitk.sitkUInt8
            #             )
            #         )
            #         # probability based index selection
            #         del prob_gm
            #         del prob_wm
            #         del prom_csf
            #
            #     except Exception as e:
            #         print("Subject {} Failed".format(sub.get_subject_id()))

        return np.array(images), np.array(labels)

    def extract_engineered_features(self, subject, train=False):
        edge_img = subject.get_edge_mask()
        edge_arr = sitk.GetArrayFromImage(edge_img).astype(np.uint8)
        t1_img = self.cleaner.median_filter(self.cleaner.intensity_rescale(subject.get_t1_image()))
        t1_arr = sitk.GetArrayFromImage(t1_img).astype(np.uint8)

        mean = sitk.MeanImageFilter()
        mean.SetRadius(5)
        t1_avg_arr = sitk.GetArrayFromImage(mean.Execute(t1_img)).astype(np.uint8)
        edge_avg_arr = sitk.GetArrayFromImage(mean.Execute(edge_img)).astype(np.uint8)

        # print('t1', t1_arr.shape)
        # print('t1-avg', t1_avg_arr.shape)
        # # print('t2', t2_arr.shape)
        # # print('t2-avg', t2_avg_arr.shape)
        # print('edge', edge_arr.shape)
        # print('edge-avg', edge_avg_arr.shape)

        t1_arr = t1_arr.ravel()
        t1_avg_arr = t1_avg_arr.ravel()
        edge_arr = edge_arr.ravel()
        edge_avg_arr = edge_avg_arr.ravel()
        features = np.column_stack((t1_arr, t1_avg_arr, edge_arr, edge_avg_arr))
        # print('features', features.shape)

        if train:
            labels = np.reshape(sitk.GetArrayFromImage(subject.get_label_mask()).astype(np.uint8).ravel(), (-1, 1))
            # print('labels', labels.shape)
            return features, labels

        return features

    def pred_to_image(self, array, subject):
        if len(array.shape) == 1:
            array = np.reshape(array, self.cleaner.IMG_SIZE)
        image = sitk.GetImageFromArray(array)
        t1_im = subject.get_t1_image()
        image = self.cleaner.resize(image, is_label_map=True, size=t1_im.GetSize())
        image.CopyInformation(t1_im())
        return image

    def to_one_hot(self, scalar_labels, new=False):
        if new:
            self.encoder = OneHotEncoder()
            self.encoder.fit(scalar_labels)

        return self.encoder.transform(scalar_labels)

    def from_one_hot(self, one_hot_labels):
        if not self.encoder:
            raise Exception("Encoder has not been fitted yet")
        return self.encoder.inverse_transform(one_hot_labels)

    def get_max_batches(self, mode):
        return floor(len(self.get_subjects(mode=mode)) / self.batch_size)


# Object for all of the preprocessing steps with default parameters
# That I have set by looking at their effects of the dataset
class Cleaner(object):
    def __init__(self):
        self.intensity_scaler = sitk.RescaleIntensityImageFilter()
        self.median = sitk.MedianImageFilter()
        self.MAX_INTENSITY = 255
        self.MIN_INTENSITY = 0
        self.MEDIAN_RADIUS = 1
        self.IMG_SIZE = (128, 128, 128)

    def intensity_rescale(self, image, intensity_min=None, intensity_max=None):
        # input check
        if not intensity_min:
            intensity_min = self.MIN_INTENSITY

        if not intensity_max:
            intensity_max = self.MAX_INTENSITY

        self.intensity_scaler.SetOutputMaximum(intensity_max)
        self.intensity_scaler.SetOutputMinimum(intensity_min)
        return self.intensity_scaler.Execute(image)

    def median_filter(self, image, radius=None):
        # input check
        if not radius:
            radius = self.MEDIAN_RADIUS

        self.median.SetRadius(radius)
        return self.median.Execute(image)

    def mean_filter(self, image, radius=None):
        pass

    def resize(self, image, is_label_map, size=None):
        # input check
        if not size:
            size = self.IMG_SIZE

        # set interpolation based on label_map_type
        interpolator = sitk.sitkLinear
        if is_label_map:
            interpolator = sitk.sitkNearestNeighbor

        # calculate new spacing
        input_size = np.array(image.GetSize())
        input_spacing = np.array(image.GetSpacing())
        output_size = np.array(size)
        output_spacing = (input_size * input_spacing) / output_size

        # create reference image
        ref_image = sitk.Image(size[0], size[1], size[2], image.GetPixelID())
        transform = sitk.Transform(len(output_size), sitk.sitkIdentity)
        ref_image.SetOrigin(image.GetOrigin())
        ref_image.SetDirection(image.GetDirection())
        ref_image.SetSpacing(output_spacing)

        return sitk.Resample(image, ref_image, transform, interpolator)

    def clean(self, image, is_label_map, intensity_min=None, intensity_max=None, median_radius=None, size=None):
        clean = self.intensity_rescale(image=image, intensity_min=intensity_min, intensity_max=intensity_max)
        clean = self.median_filter(image=clean, radius=median_radius)
        clean = self.resize(image=clean, is_label_map=is_label_map, size=size)
        return clean


# this was a bit of a stretch goal
# class Augmenter(object):
#     def __init__(self):
#         pass

# this class includes logic for retrieving the label mask and edge image
class Subject(object):
    """
    Subject is a class that is intended to encapsulate
    all the necessary information related to managing files
    from a single subject.
    """

    def __init__(self, subject_id):
        self.subject_id = subject_id
        # T1 and other T1 file path
        self.t1_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                  'TissueClassify', 't1_average_BRAINSABC.nii.gz')
        self.t2_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                  'TissueClassify', 't2_average_BRAINSABC.nii.gz')
        # Probability map file paths
        self.gm_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                  'ACCUMULATED_POSTERIORS', 'POSTERIOR_GM_TOTAL.nii.gz')
        self.wm_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                  'ACCUMULATED_POSTERIORS', 'POSTERIOR_WM_TOTAL.nii.gz')
        self.csf_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                   'ACCUMULATED_POSTERIORS', 'POSTERIOR_CSF_TOTAL.nii.gz')
        self.bg_fn = os.path.join(GLB_DATA_DIR, self.subject_id, self.subject_id + "_hcpy1",
                                  'ACCUMULATED_POSTERIORS', 'POSTERIOR_BACKGROUND_TOTAL.nii.gz')

        # self.label_mask = None

        # config
        self.PIXEL_TYPE = sitk.sitkFloat32

    # overloading how subjects print
    def __str__(self):
        return "Subject #{0}".format(self.subject_id)

    def __repr__(self):
        return self.__str__()

    # getter for subject id
    def get_subject_id(self):
        return self.subject_id

    # getters for filename paths
    def get_t1_filename(self) -> str:
        return self.t1_fn

    def get_t2_filename(self) -> str:
        return self.t2_fn

    def get_gm_filename(self) -> str:
        return self.gm_fn

    def get_wm_filename(self) -> str:
        return self.wm_fn

    def get_csf_filename(self) -> str:
        return self.csf_fn

    def get_bg_filename(self) -> str:
        return self.bg_fn

    # utility function to wrap getting an image (in case this needs to switch to sitk)
    def get_image(self, fname: str):
        # print(fname)
        return sitk.ReadImage(fname, self.PIXEL_TYPE)

    # getters for images
    def get_t1_image(self):
        return self.get_image(self.get_t1_filename())

    def get_t2_image(self):
        return self.get_image(self.get_t2_filename())

    def get_gm_prob_mask(self):
        return self.get_image(self.get_gm_filename())

    def get_wm_prob_mask(self):
        return self.get_image(self.get_wm_filename())

    def get_csf_prob_mask(self):
        return self.get_image(self.get_csf_filename())

    def get_bg_prob_mask(self):
        return self.get_image(self.get_bg_filename())

    def get_edge_mask(self, upper=20, lower=5, variance=1):
        cleaner = Cleaner()
        clean = cleaner.intensity_rescale(self.get_t1_image())
        clean = cleaner.median_filter(clean)
        return sitk.CannyEdgeDetection(clean, lowerThreshold=lower, upperThreshold=upper,
                                       variance=[variance, variance, variance])

    # generate a label mask of the image based on the probabilities
    def get_label_mask(self):
        """
        OUTPUTS:
            label_mask where
                background -> 0
                gm -> 1
                wm -> 2
                csf -> 3

        Processing:
            - highest probability wins the voxel
            - background probability mask will be used to exclude extraneous labels
        """
        # if the label mask as been computed, don't recompute
        # if self.label_mask:
        # return self.label_mask

        # threshold of acceptable background probability
        bg_thresh = 0.0001  # less that 1/100 % probability of background

        # get numpy array versions of all probability masks as floating point
        def get_im_as_float_arr(im):
            return sitk.GetArrayFromImage(im).astype(np.float)

        gm = get_im_as_float_arr(self.get_gm_prob_mask())
        wm = get_im_as_float_arr(self.get_wm_prob_mask())
        csf = get_im_as_float_arr(self.get_csf_prob_mask())
        bg = get_im_as_float_arr(self.get_bg_prob_mask())

        # mask where grey matter has the highest probability
        gm_mask = sitk.GetImageFromArray((gm > 0.5) * 1)
        # gm has a label value of 1

        # mask where white matter has the highest probability
        wm_mask = sitk.GetImageFromArray((wm > 0.5) * 2)
        # wm has a label value of 2

        # mask where csf has the highest probability
        csf_mask = sitk.GetImageFromArray((csf > 0.5) * 3)
        # csf has a label value of 3

        # copy over information using t1 image
        t1_im = self.get_t1_image()
        gm_mask.CopyInformation(t1_im)
        wm_mask.CopyInformation(t1_im)
        csf_mask.CopyInformation(t1_im)

        # masks will not overlap because we created them based on the
        # highest probability values
        #         self.label_mask = sitk.Cast(gm_mask + wm_mask + csf_mask, sitk.sitkUInt8)
        #         return self.label_mask
        return sitk.Cast(gm_mask + wm_mask + csf_mask, sitk.sitkUInt8)


# Use enumerations to represent the various evaluation measures
class OverlapMeasures(Enum):
    jaccard, dice, false_negative, false_positive = range(4)


# utility class for learning about segmentation quality
class Evaluator(object):
    # taken from
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    # just a wrapper around the sklearn method
    @staticmethod
    def classification_report(y_true, y_pred, labels=None):
        return classification_report(y_true=y_true, y_pred=y_pred, labels=labels)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    @staticmethod
    def display_with_overlay(segmentation_number, slice_number, image, segs):
        img = image[:, :, slice_number]
        msk = segs[segmentation_number][:, :, slice_number]
        overlay_img = sitk.LabelMapContourOverlay(
            sitk.Cast(msk, sitk.sitkLabelUInt8),
            sitk.Cast(img, sitk.sitkUInt8),
            opacity=1,
            contourThickness=[2, 2])
        # We assume the original slice is isotropic, otherwise the display would be distorted
        plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
        plt.axis('off')
        plt.show()

    @staticmethod
    def get_segmentation_stats(ground_truth_segmentations, predicted_segmentations, graph=True, latex=True,
                               existing_df=None):
        assert len(ground_truth_segmentations) == len(predicted_segmentations)

        overlap_results = np.zeros((len(ground_truth_segmentations), len(OverlapMeasures.__members__.items())))
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

        for i, seg in enumerate(ground_truth_segmentations):
            # Overlap measures
            overlap_measures_filter.Execute(predicted_segmentations[i], seg)
            overlap_results[i, OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[i, OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[i, OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[i, OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()

        overlap_results_df = pd.DataFrame(data=overlap_results, index=list(range(len(ground_truth_segmentations))),
                                          columns=[name for name, _ in OverlapMeasures.__members__.items()])

        if latex:
            print(overlap_results_df.to_latex())
        if graph:
            overlap_results_df.plot(kind='bar').legend(bbox_to_anchor=(1.6, 0.9))
            plt.show()
        return overlap_results_df


# good ole random forest classifier
with open('RF.pickle', 'rb') as f:
    CLF = pickle.load(f)


def estimate_gray_white_csf(t1_image, t2_image):
    """
    This function must have exactly the input/output signature as specified
    in order to compete for extra credit.

    The estimate_grey_white_csf function applies your trained machine learning
    algorithm to the t1 & t2 image to geneate a 3 class tissue classified image.
    INPUTS:
        t1_image an itk.Image representing the t1 intensity data
        t2_image an itk.Image representing the t2 intensity data
    RETURNS
        gray_matter_image    (an itk.Image where values > 0.5 indicates gray matter)
        white_matter_image   (an itk.Image where values > 0.5 indicates white matter)
        csf_matter_image     (an itk.Image where values > 0.5 indicates csf)
    """

    # the classes and function sI designed work on subjects,
    # so I will be copying some code down just for this function to work
    # SORRY!
    cleaner = Cleaner()
    clean_t1 = cleaner.intensity_rescale(t1_image)
    clean_t1 = cleaner.median_filter(clean_t1)

    edge_img = sitk.CannyEdgeDetection(clean_t1, lowerThreshold=5, upperThreshold=20, variance=[1, 1, 1])

    edge_arr_save = sitk.GetArrayFromImage(edge_img).astype(np.uint8)
    t1_arr_save = sitk.GetArrayFromImage(clean_t1).astype(np.uint8)

    mean = sitk.MeanImageFilter()
    mean.SetRadius(5)
    t1_avg_arr = sitk.GetArrayFromImage(mean.Execute(clean_t1)).astype(np.uint8)
    edge_avg_arr = sitk.GetArrayFromImage(mean.Execute(edge_img)).astype(np.uint8)

    t1_arr = t1_arr_save.ravel()
    t1_avg_arr = t1_avg_arr.ravel()
    edge_arr = edge_arr_save.ravel()
    edge_avg_arr = edge_avg_arr.ravel()
    features = np.column_stack((t1_arr, t1_avg_arr, edge_arr, edge_avg_arr))

    prediction = CLF.predict(features)
    prediction_arr = np.reshape(prediction, t1_arr_save.shape)
    prediction_arr = np.round(prediction_arr).astype(np.uint8)

    prediction_img = sitk.GetImageFromArray(prediction_arr)
    prediction_img.CopyInformation(clean_t1)

    return prediction_img == 1, prediction_img == 2, prediction_img == 3

# Initial Deep Learning attempt
# # load the model before, so we don't have to do this for every classification
# MODEL = keras.Sequential()
#
# # input_layer
# MODEL.add(keras.layers.Conv3D(filters=32, kernel_size=3, padding='same', data_format='channels_last',
#                               input_shape=INPUT_SHAPE[1:]))
# # a bunch of conv layers
# for i in range(3):
#     MODEL.add(
#         keras.layers.Conv3D(filters=32, kernel_size=3, padding='same', data_format='channels_last', activation='relu'))
#     MODEL.add(keras.layers.Dropout(0.25))
# MODEL.add(keras.layers.Conv3D(filters=1, kernel_size=3, padding='same', data_format='channels_last'))
# MODEL.compile(
#     optimizer='adam',
#     loss='mse',
#     metrics=['accuracy', keras.metrics.MAE],
# )
#
#
# # Initial deep learning attempt
# def estimate_gray_white_csf(t1_image, t2_image):
#     """
#     This function must have exactly the input/output signature as specified
#     in order to compete for extra credit.
#
#     The estimate_grey_white_csf function applies your trained machine learning
#     algorithm to the t1 & t2 image to geneate a 3 class tissue classified image.
#     INPUTS:
#         t1_image an itk.Image representing the t1 intensity data
#         t2_image an itk.Image representing the t2 intensity data
#     RETURNS
#         gray_matter_image    (an itk.Image where values > 0.5 indicates gray matter)
#         white_matter_image   (an itk.Image where values > 0.5 indicates white matter)
#         csf_matter_image     (an itk.Image where values > 0.5 indicates csf)
#     """
#     cleaner = Cleaner()
#     im = sitk.GetArrayFromImage(
#         sitk.Cast(
#             cleaner.clean(t1_image, is_label_map=False), sitk.sitkUInt8
#         )
#     )
#     im = np.expand_dims(im, 3)
#     im = np.expand_dims(im, 0)
#
#     prediction = MODEL.predict(im)
#     prediction = np.round(prediction)
#     prediction[prediction > 3] = 3
#     prediction[prediction < 0] = 0
#
#     pred_image = sitk.GetImageFromArray(prediction[0, :, :, :, 0].astype(np.uint8))
#     pred_image = cleaner.resize(image=pred_image, is_label_map=True, size=t1_image.GetSize())
#     pred_image = sitk.Cast(pred_image, sitk.sitkUInt8)
#     # THESE ARE MEDICAL IMAGES, THEY DO NOT OVERLAP IF YOU DON'T COPY THE INFO!!! I GOOFED THIS AT FIRST
#     pred_image.CopyInformation(t1_image)
#
#     return pred_image == 1, pred_image == 2, pred_image == 3
