#Importing suitable libraries
import numpy as np
from cv2 import imread,createCLAHE,imwrite
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Data preprocessing and Image Segmentation
def image_preprocessing(img):
    imwrite('t.jpg',img)
    image = imread('t.jpg',0)
    clahe_object = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    histogram_output = clahe_object.apply(image)
    img2=histogram_output.reshape((-1,1))
    gmm_model=GMM(n_components=3,covariance_type='tied').fit(img2)
    gmm_labels=gmm_model.predict(img2)      
    segmented=gmm_labels.reshape(histogram_output.shape[0],histogram_output.shape[1])*85
    imwrite('t2.jpg',segmented)
    segmented = imread('t2.jpg')
    return segmented
#Image Augementation,Building Model,Classification of model
data=ImageDataGenerator(validation_split=0.3,vertical_flip=True,horizontal_flip=True)
path="dataset/train"
path2="dataset/test"

training=data.flow_from_directory(path,
                                  target_size=(315, 315),
                                  subset='training'
                                  )
validation=data.flow_from_directory(path,
                                 target_size=(315, 315),
                                  subset='validation')
data2=ImageDataGenerator()
test=data2.flow_from_directory(path2,
                                 target_size=(315, 315),
                                  subset='training'
                                 )



model=Sequential([
    Conv2D(filters=64,kernel_size=3,input_shape=(315,315,3)),
   MaxPooling2D(pool_size=3),
   Conv2D(filters=128,kernel_size=3,activation="relu"),
   MaxPooling2D(pool_size=3),
   Flatten(),
   Dense(500,activation='relu'),
   Dense(200,activation='relu'),
   Dropout(0.2),
   Dense(2,activation='softmax')    
    ])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history2=model.fit(training,validation_data=validation,epochs=7, steps_per_epoch=len(training)
                   , validation_steps=len(validation),batch_size=8 )

#Evaluating Model
r=model.evaluate(training)
r=model.evaluate(validation)
r2=model.evaluate(test)
#Plotting Accuracy
plt.plot(history2.history['accuracy'],label='training',c='black')
plt.plot(history2.history['val_accuracy'],label='validation',c='green')
plt.legend()
plt.ylim(0,1)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
#Plotting Loss
plt.plot(history2.history['loss'],label='training',c='black')
plt.plot(history2.history['val_loss'],label='validation',c='green')
plt.legend()
plt.ylim(0,1)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()
#Predicting Model
m=['no','yes']
import cv2
img=imread('y11.jpg')
img=cv2.resize(img,(315,315))
img=img.reshape(1,315,315,3)
x=model.predict(img)
print(m[np.argmax(x)])