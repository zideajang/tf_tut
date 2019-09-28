import tensorflow as tf
from tensorflow import keras
import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = keras.Sequential()
model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu",input_shape=X.shape[1:]))
# model.add(Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu"))
# model.add(Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))

model.add(keras.layers.Dense(1,activation="sigmoid"))
# model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=3,validation_split=0.1)
