#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


# In[2]:


DIRECTORY = r'C:\Users\Kevin\Desktop\MLPRACTIVE\Training'

CATEGORIES = ['female', 'male']


# In[3]:


data = []
img_size=60

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path)
        img_arr = cv2.resize( arr , (img_size,img_size))
        data.append([img_arr, label])
        


# In[4]:


random.shuffle(data)


# In[5]:


x=[]
y=[]

for img,idx in data:
    x.append(img)
    y.append(idx)
    
x=np.array(x)
x=x/255
y=np.array(y)


# In[6]:


y.shape


# In[7]:


model = keras.Sequential([
    layers.Conv2D(4, (4,4), activation='relu', input_shape=(60, 60, 3)),
    layers.MaxPooling2D((2,2)),

   

   
    layers.Flatten(),
    layers.Dense(25, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[8]:


model.fit(x,y,epochs=5,validation_split=.1)


# In[9]:


DIRECTORY = r'C:\Users\Kevin\Desktop\MLPRACTIVE\Validation'

CATEGORIES = ['female', 'male']

data = []
img_size=60

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = CATEGORIES.index(category)
        arr = cv2.imread(img_path)
        img_arr = cv2.resize( arr , (img_size,img_size))
        data.append([img_arr, label])
        
random.shuffle(data)

x=[]
y=[]

for img,idx in data:
    x.append(img)
    y.append(idx)
    
x=np.array(x)
x=x/255
y=np.array(y)


# In[10]:


model.evaluate(x,y)


# In[11]:


pred=model.predict(x)
print(y)


# In[12]:


type(y)


# In[13]:


print(pred)


# In[14]:


type(pred)


# In[15]:


pred2=[]
pred3=[]
for i in pred:
    pred2.append(max(i))
for i in pred2:
    if i>.5:
        pred3.append(1)
    else:
        pred3.append(0)
pred3


# In[16]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix




plot_confusion_matrix(model, x, y)
plt.show()


# In[ ]:




