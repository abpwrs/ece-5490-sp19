import numpy as np
import pandas as pd  # pip install pandas
from sklearn.naive_bayes import GaussianNB  # pip install sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
# from sklearn.model_selection.cross_validation import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import gridspec


def niave_gaussian_probabilities(inputs_test, inputs_train, lbl_np1d_train):
    """Trains a niave gaussian model.
    
    Args:
      inputs_test (np.array): Training dataset
      inputs_train (np.array): Training dataset
      lbl_np1d_train (np.array): Training labels
    Returns:
      all_probs (np.array): Estimated probabilites of each sample for each class, shape (n_samples, n_classes)
      model: Trained model 
    
    """
    gnb = GaussianNB()
    model = gnb.fit(inputs_train, lbl_np1d_train)
    all_probs = model.predict_proba(inputs_test)
    return all_probs, model


def knn_probabilities(inputs_test, inputs_train, lbl_np1d_train, k):
    """Trains a KNN model.
    
    Args:
      inputs_test (np.array): Testing dataset 
      inputs_train (np.array): Training dataset
      lbl_np1d_train (np.array): Training labels
      k (uint): Number of neighbors
    Returns:
      all_probs (np.array): Estimated probabilites of each sample for each class, shape (n_samples, n_classes)
      model Trained model
    
    """
    ###############################
    ########## FILL IN ############
    ###############################
    knn = KNeighborsClassifier(n_neighbors=k)
    model = knn.fit(inputs_train, lbl_np1d_train)
    all_probs = model.predict_proba(inputs_test)
    ###############################

    return all_probs, model


def dt_probabilities(inputs_test, inputs_train, lbl_np1d_train, maxDepth=None):
    """Trains a decision tree model.
    
    Args:
      inputs_test (np.array): Testing dataset 
      inputs_train (np.array): Training dataset
      lbl_np1d_train (np.array): Training labels
    Returns:
      all_probs (np.array): Estimated probabilites of each sample for each class, shape (n_samples, n_classes)
      model Trained model
    
    """
    dt = tree.DecisionTreeClassifier(max_depth=maxDepth)
    model = dt.fit(inputs_train, lbl_np1d_train)
    all_probs = model.predict_proba(inputs_test)
    return all_probs, model


def get_training_data(lbl_np1d, full_input_data):
    """Creates training and testing datasets from input image and mask.
    
    Args:
      lbl_np1d (np.array): mask of desired training data 
      full_input_data (np.array): all of the image data
    Returns:
      full_input_data (np.array): testing data
      train_subset (np.array): training data
    
    """
    train_subset = full_input_data[lbl_np1d > 0]
    # full_input_data[lbl_np1d < 1] = 0  # Set AIR to code 0

    return full_input_data, train_subset


def create_train_test_arrays(feature_images):
    """Combines all features to create train and test datasets.
    
    Args:
      feature_images (list): Each list element is tuple of np.arrys for training and testing data for 1 feature
    Returns:
      inputs_test (np.array): Testing dataset 
      inputs_train (np.array): Training dataset 
    
    """

    test_set = list()
    train_set = list()
    for tup in feature_images:
        test_set.append(tup[0].T)
        train_set.append(tup[1].T)
    inputs_test = np.dstack(tuple(test_set))
    # remove 0 dimension
    inputs_test.shape = (inputs_test.shape[1], inputs_test.shape[2])
    inputs_train = np.dstack(tuple(train_set))
    # remove 0 dimension
    inputs_train.shape = (inputs_train.shape[1], inputs_train.shape[2])
    return inputs_test, inputs_train


def plot_decision_boundary(model, feat1_np1d, feat2_np1d, label_np1d, class_labels):
    """Plots the decision boundary for the classifier
    
    Args:
      model : A trained model using sklearn modules
      feat1_np1d (np.array) : training data, first feaure
      feat2_np1d (np.array) : training data, second feature
      label_np1d (np.array) : training data labels
 
    """
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.1, hspace=0.1)

    # create grid of entire feature space
    x_min, x_max = feat1_np1d.min() - 1, feat1_np1d.max() + 1
    y_min, y_max = feat2_np1d.min() - 1, feat2_np1d.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # class label colors for scatter points 
    colors = {'WM': 'w', 'GM': 'c', 'CSF': 'k'}
    labels = ['WM', 'GM', 'CSF']
    # plot probability map over feature space for each class
    for ind, label in enumerate(labels):
        ax = plt.subplot(gs[ind])
        pred = Z[:, ind]
        pred = pred.reshape(xx.shape)

        # plot probability contours
        v = np.linspace(0.0, 1.0, 7, endpoint=True)
        cont = ax.contourf(xx, yy, pred, v, cmap=plt.cm.RdBu, alpha=0.6)
        ax.contour(xx, yy, pred, levels=[0.5], colors='black')

        # plot points for each class
        for ind2, label2 in enumerate(labels):
            xc = [val for i, val in enumerate(feat1_np1d) if label_np1d[i] == class_labels[label2]]
            yc = [val for i, val in enumerate(feat2_np1d) if label_np1d[i] == class_labels[label2]]
            cols = colors[label2] * len(xc)
            ax.scatter(xc, yc, c=cols, label=label2)

        ax.set_title(label + ' Probability')
        ax.set_xlabel('T1')
        ax.set_ylabel('T2')
        ax.legend()
        fig.colorbar(cont, shrink=0.9, ticks=v)
    plt.show()


def get_random_sample(feat1_np1d, feat2_np1d, label_np1d, LABELS, num_per_class=100):
    """
    """
    feat1 = []
    feat2 = []
    feat1_train = np.array([])
    feat2_train = np.array([])
    label_train = np.array([])
    for label in LABELS:
        value = LABELS[label]
        feat1 = feat1_np1d[label_np1d == value]
        feat2 = feat2_np1d[label_np1d == value]
        ind = [x for x in range(0, feat1.shape[0])]
        np.random.shuffle(ind)
        ind_sub = ind[0:num_per_class]
        feat1_train = np.hstack((feat1_train, feat1[ind_sub]))
        feat2_train = np.hstack((feat2_train, feat2[ind_sub]))
        label_train = np.hstack((label_train, value * np.ones(num_per_class)))
    train = np.column_stack((feat1_train, feat2_train))
    return train, label_train


def get_accuracy(pred, lbl):
    """
    Compute the accuracy of the values based on the machine learning model
    """
    correct = sum((np.logical_and(pred == lbl, lbl > 0)).astype(int))
    total = sum((lbl > 0).astype(int))
    return float(correct) / total
