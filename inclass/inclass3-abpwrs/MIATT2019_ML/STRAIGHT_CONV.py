
# coding: utf-8

# In[1]:


from tensorflow import keras
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from HW8functions import Subject, HCPData, Cleaner, ModeKeys, plot_confusion_matrix, CLASS_NAMES
INPUT_SHAPE = [None, 128, 128, 128, 1]


# In[2]:


dataset = HCPData()


# In[ ]:


dataset.batch_size = 100
dataset.init_batches(ModeKeys.TRAIN)
train_images, train_labels = dataset.next_batch()


# In[ ]:


dataset.batch_size = 10
dataset.init_batches(ModeKeys.VAL)
val_images, val_labels = dataset.next_batch()


# In[ ]:


train_images.shape, train_labels.shape


# In[3]:


model = keras.Sequential()

# input_layer
model.add(keras.layers.Conv3D(filters=32, kernel_size=3, padding='same', data_format='channels_last', input_shape=INPUT_SHAPE[1:]))
# a bunch of conv layers
for i in range(3):
    model.add(keras.layers.Conv3D(filters=32, kernel_size=3, padding='same', data_format='channels_last', activation='relu'))
    model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv3D(filters=1, kernel_size=3, padding='same', data_format='channels_last'))
model.summary()


# In[ ]:


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy', keras.metrics.MAE],
)


# In[35]:


model_dir = os.path.join('/localscratch/Users/abpwrs', "models","STRAIGHT_CONV_20190528")
tboard_dir = os.path.join('/localscratch/Users/abpwrs', 'logs','STRAIGHT_CONV_20190528')
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.isdir(tboard_dir):
    os.mkdir(tboard_dir)

model_checkpoint = ModelCheckpoint(os.path.join(model_dir,"weights.{epoch:02d}-{val_loss:.2f}.hdf5"), save_best_only=True)
tensor_board = TensorBoard(log_dir=tboard_dir, write_graph=True, write_images=True)


# In[36]:


model.fit(
    x=train_images,
    y=train_labels,
    batch_size=1,
    verbose=1,
    epochs=1000,
    validation_data=(val_images, val_labels),
    callbacks=[model_checkpoint, tensor_board],
    use_multiprocessing=False
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




