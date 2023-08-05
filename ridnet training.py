import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import cv2 
import numpy as np 
import pickle

###############################################################################################
# load image
###############################################################################################
f = open(r'noise and referance (64x64).pckl', 'rb') 
data_in, data_out = pickle.load(f)
f.close()

## normalize ##
x_train = np.float32(data_in)/255
y_train = np.float32(data_out)/255

## random shuffle traning data (optional) ##
# seed = np.random.randint(123)
# np.random.seed(seed)
# np.random.shuffle(x_train)
# np.random.seed(seed)
# np.random.shuffle(y_train)

###############################################################################################
# Enhancement Attention Modules (EAM)
###############################################################################################
# class EAM(tf.keras.layers.Layer):
# 	def __init__(self,**kwargs):
# 		super().__init__(**kwargs)

# 		self.conv1 = layers.Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')
# 		self.conv2 = layers.Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu') 

# 		self.conv3 = layers.Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')
# 		self.conv4 = layers.Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')

# 		self.conv5 = layers.Conv2D(64, (3,3),padding='same',activation='relu')

# 		self.conv6 = layers.Conv2D(64, (3,3),padding='same',activation='relu')
# 		self.conv7 = layers.Conv2D(64, (3,3),padding='same')

# 		self.conv8 = layers.Conv2D(64, (3,3),padding='same',activation='relu')
# 		self.conv9 = layers.Conv2D(64, (3,3),padding='same',activation='relu')
# 		self.conv10 = layers.Conv2D(64, (1,1),padding='same')

# 		self.gap = layers.GlobalAveragePooling2D()

# 		self.conv11 = layers.Conv2D(64, (3,3),padding='same',activation='relu')
# 		self.conv12 = layers.Conv2D(64, (3,3),padding='same',activation='sigmoid')

# 	def call(self,input):
# 		conv1 = self.conv1(input)
# 		conv1 = self.conv2(conv1)

# 		conv2 = self.conv3(input)
# 		conv2 = self.conv4(conv2)

# 		concat = layers.concatenate([conv1,conv2])
# 		conv3 = self.conv5(concat)
# 		add1 = layers.Add()([input,conv3])

# 		conv4 = self.conv6(add1)
# 		conv4 = self.conv7(conv4)
# 		add2 = layers.Add()([conv4,add1])
# 		add2 = layers.Activation('relu')(add2)

# 		conv5 = self.conv8(add2)
# 		conv5 = self.conv9(conv5)
# 		conv5 = self.conv10(conv5)
# 		add3 = layers.Add()([add2,conv5])
# 		add3 = layers.Activation('relu')(add3)

# 		gap = self.gap(add3)
# 		gap = layers.Reshape((1,1,64))(gap)
# 		conv6 = self.conv11(gap)
# 		conv6 = self.conv12(conv6)

# 		mul = layers.Multiply()([conv6, add3])
# 		out = layers.Add()([input,mul]) # This is not included in the reference code
# 		return out

def EAM(input):
		conv1 = layers.Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')(input)
		conv1 = layers.Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu')(conv1)

		conv2 = layers.Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')(input)
		conv2 = layers.Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')(conv2)

		concat = layers.concatenate([conv1,conv2])
		conv3 = layers.Conv2D(64, (3,3),padding='same',activation='relu')(concat)
		add1 = layers.Add()([input,conv3])

		conv4 = layers.Conv2D(64, (3,3),padding='same',activation='relu')(add1)
		conv4 = layers.Conv2D(64, (3,3),padding='same')(conv4)
		add2 = layers.Add()([conv4,add1])
		add2 = layers.Activation('relu')(add2)

		conv5 = layers.Conv2D(64, (3,3),padding='same',activation='relu')(add2)
		conv5 = layers.Conv2D(64, (3,3),padding='same',activation='relu')(conv5)
		conv5 = layers.Conv2D(64, (1,1),padding='same')(conv5)
		add3 = layers.Add()([add2,conv5])
		add3 = layers.Activation('relu')(add3)

		gap = layers.GlobalAveragePooling2D()(add3)
		gap = layers.Reshape((1,1,64))(gap)
		conv6 = layers.Conv2D(64, (3,3),padding='same',activation='relu')(gap)
		conv6 = layers.Conv2D(64, (3,3),padding='same',activation='sigmoid')(conv6)

		mul = layers.Multiply()([conv6, add3])
		out = layers.Add()([input,mul]) # This is not included in the reference code
		return out

###############################################################################################
# RIDnet autoencoder (https://medium.com/analytics-vidhya/image-denoising-using-deep-learning-dc2b19a3fd54)
###############################################################################################
#### RIDnet layers ####
tf.keras.backend.clear_session()
input = keras.Input(shape=(64, 64, 3))
conv1 = layers.Conv2D(64 , (3,3),padding='same')(input)
eam1 = EAM(conv1)
eam2 = EAM(eam1)
eam3 = EAM(eam2)
eam4 = EAM(eam3)
conv2 = layers.Conv2D(3, (3,3),padding='same')(eam4) 
output = layers.Add()([input,conv2])
#### RIDnet layers ####

RIDNet = keras.Model(input,output)

###############################################################################################
# loss function
###############################################################################################
RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError())

###############################################################################################
# Training
###############################################################################################
print(RIDNet.summary())
check_point = tf.keras.callbacks.ModelCheckpoint('RIDNet.h5', monitor='val_loss')
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5) 
RIDNet.fit(x_train, y_train, epochs=15, validation_split=0.1, callbacks=[check_point, early_stopping])

print('done')