#!/usr/bin/env python
# coding: utf-8

# ## Import tensorflow and numpy

# In[1]:


import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ## Format the train data to be in a 16x16 numpy array

# In[2]:


import pathlib
data_dir = pathlib.Path("train-data.txt")
train_ds = np.loadtxt(fname=data_dir)

train_ds = np.hsplit(train_ds,[1])

x_train = train_ds[1].reshape(train_ds[1].shape[0],16,16)
y_train = train_ds[0].reshape(train_ds[0].shape[0])


# ## Format the test data to be in a 16x16 numpy array

# In[3]:


test_dir = pathlib.Path("test-data.txt")
test_ds = np.loadtxt(fname=test_dir)

test_ds = np.hsplit(test_ds,[1])

x_test = test_ds[1].reshape(test_ds[1].shape[0],16,16)
y_test = test_ds[0].reshape(test_ds[0].shape[0])


# ## Create our machine learning model

# In[4]:


num_classes = 10

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1, input_shape=(16,16,1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


# ## Configure our model for training

# In[5]:


model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


# ## Train our machine learning model

# In[6]:


epochs=15
history = model.fit(
    x=x_train,y=y_train,
    validation_data=(x_test,y_test),
    epochs=epochs
)


# ## Display our classification report and confusion matrix

# In[7]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
y_pred = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

