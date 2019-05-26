import glob
import os
from PIL import Image, ImageDraw
import numpy as np

path_out_train = '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/cla_test/train/'
path_out_test = '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/cla_test/test/'

x_train = []
x_test = []

y_train_len = []
y_test_len = []

y_train_wid = []
y_test_wid = []

y_train_ang = []
y_test_ang = []

y_train_col = []
y_test_col = []


for train_file in glob.glob(path_out_train+'*.jpg') :
    img_x_train = Image.open(train_file);
    x_train.append(np.array(img_x_train))
    image_name = str(train_file).split('/')[-1]
    image_name_initials = image_name.split('.')[0]
    #class_no = (int(image_name_initials.split('_')[0])*48)+(int(image_name_initials.split('_')[1])*24)+(int(image_name_initials.split('_')[2])*2)+(int(image_name_initials.split('_')[3]))
    #y_train.append(class_no)
    y_train_len.append(int(image_name_initials.split('_')[0]))
    y_train_wid.append(int(image_name_initials.split('_')[1]))
    y_train_ang.append(int(image_name_initials.split('_')[2]))
    y_train_col.append(int(image_name_initials.split('_')[3]))
    img_x_train.close()
    
for test_file in glob.glob(path_out_test+'*.jpg') :
    img_x_test = Image.open(test_file);
    x_test.append(np.array(img_x_test))
    image_name = str(test_file).split('/')[-1]
    image_name_initials = image_name.split('.')[0]
    #class_no = (int(image_name_initials.split('_')[0])*48)+(int(image_name_initials.split('_')[1])*24)+(int(image_name_initials.split('_')[2])*2)+(int(image_name_initials.split('_')[3]))
    #y_test.append(class_no)
    y_test_len.append(int(image_name_initials.split('_')[0]))
    y_test_wid.append(int(image_name_initials.split('_')[1]))
    y_test_ang.append(int(image_name_initials.split('_')[2]))
    y_test_col.append(int(image_name_initials.split('_')[3]))
    img_x_test.close()
    
    
x_train = np.array(x_train).astype('float32')
x_test = np.array(x_test).astype('float32')
y_train_len = np.array(y_train_len)
y_test_len = np.array(y_test_len)
y_train_wid = np.array(y_train_wid)
y_test_wid = np.array(y_test_wid)
y_train_ang = np.array(y_train_ang)
y_test_ang = np.array(y_test_ang)
y_train_col = np.array(y_train_col)
y_test_col = np.array(y_test_col)


x_train /= 255
x_test /= 255


import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import layers

input_image = keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),name='input_image')# 28*28*3


from tensorflow.keras import layers

conv2d_1 = layers.Conv2D(filters=6,kernel_size=(3,3),activation=tf.nn.relu,name="conv_1")(input_image)#26*26*6
max_pooling_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2,name="max_pool_1")(conv2d_1)#13*13*6
conv2d_2 = layers.Conv2D(filters=16,kernel_size=(4,4),activation=tf.nn.relu,name="conv_2")(max_pooling_1)#10*10*16
max_pooling_2 = layers.MaxPooling2D(pool_size=(2,2),strides=2,name="max_pool_2")(conv2d_2)#5*5*16

model = keras.Model(inputs=input_image, outputs=max_pooling_2, name='CNN_feature_extractor')

model.summary()



from keras.utils import plot_model
keras.utils.plot_model(model, '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/my_model.png')



flat_feature = layers.Flatten()(max_pooling_2)

#flat_feature.summary()





len_classifier_dense_1 = layers.Dense(256, activation='relu',name='len_dense_1')(flat_feature)
len_classifier_drop = layers.Dropout(0.2,name='len_drop')(len_classifier_dense_1)
len_classifier_dense_2 = layers.Dense(128, activation='relu',name='len_dense_2')(len_classifier_drop)
len_classifier_dense_3 = layers.Dense(64,activation='relu',name='len_dense_3')(len_classifier_dense_2)
len_classifier_dense_4 = layers.Dense(32,activation='relu',name='len_dense_4')(len_classifier_dense_3)
len_classifier_dense_5 = layers.Dense(2,activation='relu',name='len_dense_5')(len_classifier_dense_4)
len_classifier_out = layers.Dense(1,activation='sigmoid',name='len_out')(len_classifier_dense_5)




wid_classifier_dense_1 = layers.Dense(256, activation='relu',name='wid_dense_1')(flat_feature)
wid_classifier_drop = layers.Dropout(0.2,name='wid_drop')(wid_classifier_dense_1)
wid_classifier_dense_2 = layers.Dense(128, activation='relu',name='wid_dense_2')(wid_classifier_drop)
wid_classifier_dense_3 = layers.Dense(64,activation='relu',name='wid_dense_3')(wid_classifier_dense_2)
wid_classifier_dense_4 = layers.Dense(32,activation='relu',name='wid_dense_4')(wid_classifier_dense_3)
wid_classifier_dense_5 = layers.Dense(2,activation='relu',name='wid_dense_5')(wid_classifier_dense_4)
wid_classifier_out = layers.Dense(1,activation='sigmoid',name='wid_out')(wid_classifier_dense_5)




col_classifier_dense_1 = layers.Dense(256, activation='relu',name='col_dense_1')(flat_feature)
col_classifier_drop = layers.Dropout(0.2,name='col_drop')(col_classifier_dense_1)
col_classifier_dense_2 = layers.Dense(128, activation='relu',name='col_dense_2')(col_classifier_drop)
col_classifier_dense_3 = layers.Dense(64,activation='relu',name='col_dense_3')(col_classifier_dense_2)
col_classifier_dense_4 = layers.Dense(32,activation='relu',name='col_dense_4')(col_classifier_dense_3)
col_classifier_dense_5 = layers.Dense(2,activation='relu',name='col_dense_5')(col_classifier_dense_4)
col_classifier_out = layers.Dense(1,activation='sigmoid',name='col_out')(col_classifier_dense_5)




ang_classifier_dense_1 = layers.Dense(256, activation='relu',name='ang_dense_1')(flat_feature)
ang_classifier_drop = layers.Dropout(0.2,name='ang_drop')(ang_classifier_dense_1)
ang_classifier_dense_2 = layers.Dense(128, activation='relu',name='ang_dense_2')(ang_classifier_drop)
ang_classifier_dense_3 = layers.Dense(64,activation='relu',name='ang_dense_3')(ang_classifier_dense_2)
ang_classifier_dense_4 = layers.Dense(32,activation='relu',name='ang_dense_4')(ang_classifier_dense_3)
ang_classifier_out = layers.Dense(12,activation='softmax',name='ang_out')(ang_classifier_dense_4)





model_2 = keras.Model(inputs=input_image,outputs=[len_classifier_out,wid_classifier_out,ang_classifier_out,col_classifier_out])



model_2.summary()




keras.utils.plot_model(model_2, '/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/multi_output_model.png', show_shapes=True)



model_2.compile(optimizer='adam',loss={'len_out': 'binary_crossentropy','wid_out': 'binary_crossentropy','ang_out': 'categorical_crossentropy','col_out': 'binary_crossentropy'},loss_weights=[1.0,1.0,1.0,1.0],metrics=['acc'])


from keras.utils import to_categorical

#y_train_ang_2 = to_categorical(y_train_ang, 12)#12 classes of angles



y_train_ang = to_categorical(y_train_ang, 12)#12 classes of angles


y_test_ang = to_categorical(y_test_ang,12)#12 classes of angles


model_2.fit({'input_image':x_train},{'len_out':y_train_len, 'wid_out':y_train_wid, 'ang_out':y_train_ang, 'col_out':y_train_col},epochs=3,batch_size=32,verbose=2)


test_scores = model_2.evaluate(x=x_test,y=[y_test_len,y_test_wid,y_test_ang,y_test_col],verbose=1)

print('Test Scores :', test_scores)

from keras.utils import plot_model
plot_model(model_2, to_file='/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/multi_head_model.png')

model_2.save('/home/user/Downloads/sem_6/CS_671_DL/Assignment_2/multi_headed_model_save_epoch_3.h5')
