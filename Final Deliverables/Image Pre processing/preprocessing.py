#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('conda install --yes keras')


# In[5]:


get_ipython().system('conda install --yes tensorflow')


# In[6]:


from keras.preprocessing.image import ImageDataGenerator


# In[7]:


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


# In[8]:


test_datagen=ImageDataGenerator(rescale=1./255)


# In[9]:


x_train=train_datagen.flow_from_directory(r'C:\Users\HARIHARAN\PycharmProjects\AI Analyzer for fitness enthusiasts\TRAIN_SET',target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse')


# In[10]:


x_test=test_datagen.flow_from_directory(r'C:\Users\HARIHARAN\PycharmProjects\AI Analyzer for fitness enthusiasts\TEST_SET',target_size=(64, 64),batch_size=5,color_mode='rgb',class_mode='sparse')


# In[11]:


print(x_train.class_indices)


# In[12]:


print(x_test.class_indices)


# In[13]:


from collections import Counter as c
c(x_train.labels)


# In[ ]:




