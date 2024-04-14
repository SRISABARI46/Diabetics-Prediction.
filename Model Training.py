#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[6]:


tf.test.is_gpu_available()


# In[7]:


tf.config.list_physical_devices('GPU')


# In[2]:


BATCH_SIZE=8
EPOCHS = 5


# In[3]:


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 0.2,shear_range = 0.2,
    zoom_range = 0.2,width_shift_range = 0.2,
    height_shift_range = 0.2, validation_split = 0.2)


# In[4]:


train_data= train_datagen.flow_from_directory(r'C:\Users\ASUS\Documents\mrlEyes_2018_01\dataset\train\train_dataset',
                                target_size = (80,80), batch_size = BATCH_SIZE, 
                                class_mode = 'categorical',subset='training' )


# In[5]:


validation_data= train_datagen.flow_from_directory(r'C:\Users\ASUS\Documents\mrlEyes_2018_01\dataset\train\train_dataset',
                                target_size = (80,80), batch_size =BATCH_SIZE, 
                                class_mode = 'categorical', subset='validation')


# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[8]:


test_data = test_datagen.flow_from_directory(r'C:\Users\ASUS\Documents\mrlEyes_2018_01\dataset\train\test_dataset',
                                target_size=(80,80), batch_size = BATCH_SIZE, class_mode='categorical')


# In[9]:


bmodel = InceptionV3(include_top = False, weights = 'imagenet', 
                     input_tensor = Input(shape = (80,80,3)))


# In[10]:


hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation = 'relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2,activation = 'softmax')(hmodel)

model = Model(inputs = bmodel.input, outputs= hmodel)
for layer in bmodel.layers:
    layer.trainable = False



# In[11]:


from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau


# In[14]:


checkpoint = ModelCheckpoint(r'C:\Users\ASUS\Documents\mrlEyes_2018_01\models\model.h5',
                            monitor = 'val_loss', save_best_only = True, verbose = 3)
earlystop = EarlyStopping(monitor = 'val_loss', patience = 7, 
                          verbose= 3, restore_best_weights = True)
learning_rate = ReduceLROnPlateau(monitor= 'val_loss', patience=3, verbose= 3, )

callbacks = [checkpoint, earlystop, learning_rate]



# In[ ]:


model.compile(optimizer = 'Adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
model.fit_generator(train_data,steps_per_epoch = train_data.samples// BATCH_SIZE,
                   validation_data = validation_data,
                   validation_steps = validation_data.samples// BATCH_SIZE,
                   callbacks = callbacks,
                    epochs = EPOCHS)


# In[ ]:


acc_tr, loss_tr = model.evaluate_generator(train_data)
print(acc_tr)
print(loss_tr)


# In[ ]:


acc_vr, loss_vr = model.evaluate_generator(validation_data)
print(acc_vr)
print(loss_vr)


# In[ ]:


acc_test, loss_test = model.evaluate_generator(test_data)
print(acc_tr)
print(loss_tr)

