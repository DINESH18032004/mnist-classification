# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

MNIST Handwritten Digit Classification Dataset is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.


![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/027c58e0-ec53-48a5-b675-d165309dddc2)


## Neural Network Model

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/a5e67752-3714-4ec6-9803-6f3ddded79f3)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:

Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name: DINESH KUMAR R
### Register Number: 212222110010

```PY
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')),
model.add(layers.MaxPool2D(pool_size=(2, 2))),
model.add(layers.Flatten()),
model.add(layers.Dense(128,activation='relu')),
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/Screenshot 2024-03-11 083455.png')

type(img)

img = image.load_img('/content/Screenshot 2024-03-11 083455.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)


```


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/57fc7bec-ab25-4b12-ba54-8528b7a6fcc2)

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/f60335c0-9a7c-4336-bfd3-df2d807da3b4)


### Classification Report

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/c349bec7-4e5f-481a-a25a-6b61ab744212)


### Confusion Matrix

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/94ecd9d0-8106-4b9d-857b-6bd5810997ad)

### New Sample Data Prediction

![image](https://github.com/DINESH18032004/mnist-classification/assets/119477784/6048e818-b3fb-4a8a-b9bd-8763a42f2e64)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
