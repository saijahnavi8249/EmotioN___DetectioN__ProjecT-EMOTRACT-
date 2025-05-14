#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D , MaxPooling2D , Dropout , Flatten , Dense , BatchNormalization
from keras.callbacks import EarlyStopping
import cv2
import numpy as np
import os


# In[2]:


random = np.random.randint(0,430)
img_path_angry = []
img_path_disgusted = []
img_path_fearful = []
img_path_happy = []
img_path_neutral = []
img_path_sad = []
img_path_surprised = []
classes = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

path = r'C:\Users\jahna\archive (1)'

for j in os.walk(path):
    if classes[0] in j[0]:
        for i in j[2]:
            img_path_angry.append(fr'{j[0]}/{i}')
    elif classes[1] in j[0]:
        for i in j[2]:
            img_path_disgusted.append(fr'{j[0]}/{i}')
    elif classes[2] in j[0]:
        for i in j[2]:
            img_path_fearful.append(fr'{j[0]}/{i}')
    elif classes[3] in j[0]:
        for i in j[2]:
            img_path_happy.append(fr'{j[0]}/{i}')
    elif classes[4] in j[0]:
        for i in j[2]:
            img_path_neutral.append(fr'{j[0]}/{i}')
    elif classes[5] in j[0]:
        for i in j[2]:
            img_path_sad.append(fr'{j[0]}/{i}')
    elif classes[6] in j[0]:
        for i in j[2]:
            img_path_surprised.append(fr'{j[0]}/{i}')


# In[3]:


def plot_image():
    
    i = np.random.randint(0,100)

    img_angry = cv2.imread(img_path_angry[i])
    img_disgusted = cv2.imread(img_path_disgusted[i])
    img_fearful = cv2.imread(img_path_fearful[i])
    img_happy = cv2.imread(img_path_happy[i])
    img_neutral = cv2.imread(img_path_neutral[i])
    img_sad = cv2.imread(img_path_sad[i])
    img_surprised = cv2.imread(img_path_surprised[i])


    fig , axs = plt.subplots(1,7,figsize=[13,15])

    axs[0].imshow(img_angry)
    axs[0].set_title('Angry')
    
    axs[1].imshow(img_disgusted)
    axs[1].set_title('Disgusted')
    
    axs[2].imshow(img_fearful)
    axs[2].set_title('Fearful')

    axs[3].imshow(img_happy)
    axs[3].set_title('Happy')

    axs[4].imshow(img_neutral)
    axs[4].set_title('Neutral')

    axs[5].imshow(img_sad)
    axs[5].set_title('Sad')

    axs[6].imshow(img_surprised)
    axs[6].set_title('Suprised')


# In[4]:


for i in range(0,6):
    plot_image()



# In[5]:


data_gen = ImageDataGenerator(rescale = 1./255,
                              zoom_range = 0.2,
                              shear_range = 0.2,
                              validation_split=0.2,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              horizontal_flip=True,
)


# In[6]:


path = r'C:\Users\jahna\archive (1)'
train_dataset = data_gen.flow_from_directory(
    path,
    target_size = (48,48),
    class_mode = 'categorical',
    subset = 'training',
    color_mode = 'grayscale'
)


# In[7]:


path = r'C:\Users\jahna\archive (1)'
valid_dataset = data_gen.flow_from_directory(
    path,
    target_size = (48,48),
    class_mode = 'categorical',
    subset = 'validation',
    color_mode = 'grayscale'
)


# In[8]:


data_gen = ImageDataGenerator(rescale = 1./255,
)


# In[9]:


path = r'C:\Users\jahna\archive (1)'
test_dataset = data_gen.flow_from_directory(
    path,
    target_size = (48,48),
    class_mode = 'categorical',
    color_mode = 'grayscale'
)


# In[10]:


model = keras.Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same',kernel_regularizer='l2'))
model.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #max pooling to decrease dimension
model.add(Dropout(0.25)) #test

model.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #max pooling to decrease dimension
model.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation = 'relu',kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=keras.losses.CategoricalCrossentropy(),metrics=['acc'])


# In[13]:


early = EarlyStopping(monitor='val_loss',patience=10)


# In[14]:


model.save('model.h5')


# In[15]:


random = np.random.randint(0,100)
img_path_angry = []
img_path_disgusted = []
img_path_fearful = []
img_path_happy = []
img_path_neutral = []
img_path_sad = []
img_path_surprised = []

classes = ['angry','disgusted','fearful','happy','neutral','sad','surprised']

path = r'C:\Users\jahna\archive (1)'

for j in os.walk(path):
    if classes[0] in j[0]:
        for i in j[2]:
            img_path_angry.append(fr'{j[0]}/{i}')
    elif classes[1] in j[0]:
        for i in j[2]:
            img_path_disgusted.append(fr'{j[0]}/{i}')
    elif classes[2] in j[0]:
        for i in j[2]:
            img_path_fearful.append(fr'{j[0]}/{i}')
    elif classes[3] in j[0]:
        for i in j[2]:
            img_path_happy.append(fr'{j[0]}/{i}')
    elif classes[4] in j[0]:
        for i in j[2]:
            img_path_neutral.append(fr'{j[0]}/{i}')
    elif classes[5] in j[0]:
        for i in j[2]:
            img_path_sad.append(fr'{j[0]}/{i}')
    elif classes[6] in j[0]:
        for i in j[2]:
            img_path_surprised.append(fr'{j[0]}/{i}')


# In[16]:


def plot_pred_image():
    
    i = np.random.randint(0,100)

    

    img_angry = cv2.imread(img_path_angry[i])
    img_disgusted = cv2.imread(img_path_disgusted[i])
    img_fearful = cv2.imread(img_path_fearful[i])
    img_happy = cv2.imread(img_path_happy[i])
    img_neutral = cv2.imread(img_path_neutral[i])
    img_sad = cv2.imread(img_path_sad[i])
    img_surprised = cv2.imread(img_path_surprised[i])
    
    img_angry = cv2.cvtColor(img_angry,cv2.COLOR_BGR2GRAY)
    img_disgusted = cv2.cvtColor(img_disgusted,cv2.COLOR_BGR2GRAY)
    img_fearful = cv2.cvtColor(img_fearful,cv2.COLOR_BGR2GRAY)
    img_happy = cv2.cvtColor(img_happy,cv2.COLOR_BGR2GRAY)
    img_neutral = cv2.cvtColor(img_neutral,cv2.COLOR_BGR2GRAY)
    img_sad = cv2.cvtColor(img_sad,cv2.COLOR_BGR2GRAY)
    img_surprised = cv2.cvtColor(img_surprised,cv2.COLOR_BGR2GRAY)

    img_angry_pre = img_to_array(img_angry) / 255
    img_disgusted_pre = img_to_array(img_disgusted) / 255
    img_fearful_pre = img_to_array(img_fearful) / 255
    img_happy_pre = img_to_array(img_happy) / 255
    img_neutral_pre = img_to_array(img_neutral) / 255
    img_sad_pre = img_to_array(img_sad) / 255
    img_surprised_pre = img_to_array(img_surprised) / 255

    img_angry_pre = np.expand_dims(img_angry_pre,axis=0)
    img_disgusted_pre = np.expand_dims(img_disgusted_pre,axis=0)
    img_fearful_pre = np.expand_dims(img_fearful_pre,axis=0)
    img_happy_pre = np.expand_dims(img_happy_pre,axis=0)
    img_neutral_pre = np.expand_dims(img_neutral_pre,axis=0)
    img_sad_pre = np.expand_dims(img_sad_pre,axis=0)
    img_surprised_pre = np.expand_dims(img_surprised_pre,axis=0)

    pred_angry = model.predict(img_angry_pre)
    pred_disgusted = model.predict(img_disgusted_pre)
    pred_fearful = model.predict(img_fearful_pre)
    pred_happy = model.predict(img_happy_pre)
    pred_neutral = model.predict(img_neutral_pre)
    pred_sad = model.predict(img_sad_pre)
    pred_surprised = model.predict(img_surprised_pre)

    print(pred_angry[0])
    
    angry_title_pred = classes[pred_angry[0].argmax()]
        
    disgusted_title_pred = classes[pred_disgusted[0].argmax()]
        
    fearful_title_pred = classes[pred_fearful[0].argmax()]
        
    happy_title_pred = classes[pred_happy[0].argmax()]
        
    neutral_title_pred = classes[pred_sad[0].argmax()]
        
    sad_title_pred = classes[pred_neutral[0].argmax()]
        
    surprised_title_pred = classes[pred_surprised[0].argmax()]
    
    img_angry = cv2.cvtColor(img_angry,cv2.COLOR_GRAY2BGR)
    img_disgusted = cv2.cvtColor(img_disgusted,cv2.COLOR_GRAY2BGR)
    img_fearful = cv2.cvtColor(img_fearful,cv2.COLOR_GRAY2BGR)
    img_happy = cv2.cvtColor(img_happy,cv2.COLOR_GRAY2BGR)
    img_neutral = cv2.cvtColor(img_neutral,cv2.COLOR_GRAY2BGR)
    img_sad = cv2.cvtColor(img_sad,cv2.COLOR_GRAY2BGR)
    img_surprised = cv2.cvtColor(img_surprised,cv2.COLOR_GRAY2BGR)


    fig , axs = plt.subplots(3,3,figsize=[11,13])

    axs[0][0].imshow(img_angry)
    axs[0][0].set_title(f'Predict:{angry_title_pred}  orginal:Angry')

    axs[0][1].imshow(img_disgusted)
    axs[0][1].set_title(f'Predict:{disgusted_title_pred}  orginal:Disgusted')

    axs[0][2].imshow(img_happy)
    axs[0][2].set_title(f'Predict:{fearful_title_pred}  orginal:Happy')

    axs[1][0].imshow(img_fearful)
    axs[1][0].set_title(f'Predict:{happy_title_pred}  orginal:Fearful')

    axs[1][1].imshow(img_neutral)
    axs[1][1].set_title(f'Predict:{neutral_title_pred}  orginal:Neutral')

    axs[1][2].imshow(img_sad)
    axs[1][2].set_title(f'Predict:{sad_title_pred}  orginal:Sad')

    axs[2][1].imshow(img_surprised)
    axs[2][1].set_title(f'Predict:{surprised_title_pred}  orginal:Surprised')


# In[17]:


plot_pred_image()


# In[ ]:




