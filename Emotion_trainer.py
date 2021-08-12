import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import cv2
#from google.colab.patches import cv2_imshow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import pcl


class_names = dict([(name, cls) for name, cls in pcl.__dict__.items() if isinstance(cls, type)])
for name, cls in class_names.items():
    print(name)

train = 'data/train'
test = 'data/test'

num_train = 28707
num_val = 7174
batch_size = 32
num_epoch = 20

data_train = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

valid_gen = val_datagen.flow_from_directory(
        test,
        batch_size=batch_size,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

training_gen = data_train.flow_from_directory(
        train,
        batch_size=batch_size,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

# Model Creation
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Computing loss and saving weights
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model_info = model.fit_generator(
training_gen,
validation_data=valid_gen,
epochs=num_epoch,
validation_steps =num_val // batch_size)
model.save_weights('model.h5')
