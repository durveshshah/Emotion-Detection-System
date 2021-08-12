import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,BatchNormalization
import cv2
#from google.colab.patches import cv2_imshow
from tensorflow.keras.layers import MaxPooling2D,Activation

dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


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

model.load_weights('model.h5')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    harr_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_faces = harr_classifier.detectMultiScale(grayscale)

    for (x, y, w, h) in detect_faces:
        cv2.rectangle(frame, (x, y-43), (x+w, y+h+12), (255, 0, 0), 2)
        gray = grayscale[y:y + h, x:x + w]
        cutted_image = np.expand_dims(np.expand_dims(cv2.resize(gray, (42, 42)), -2), 1)
        pred = model.predict(cutted_image)
        ind = int(np.argmax(pred))
        cv2.putText(frame, dict[ind], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()