import cv2
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
#from google.colab.patches import cv2_imshow
from tensorflow.keras.layers import Conv2D

dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Model Initialization
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
    ret, img = cap.read()
    if not ret:
        break
    harr_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detect_faces = harr_classifier.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    for (x, y, w, h) in detect_faces:
        cv2.rectangle(img, (x, y-43), (x+w, y+h+12), (255, 0, 0), 2)
        ind = int(np.argmax(model.predict(np.expand_dims(np.expand_dims(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w], (42, 42)), -2), 1))))
        cv2.putText(img, dict[ind], (x+20, y-60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1,cv2.LINE_8)

    cv2.imshow('Video', cv2.resize(img,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

'''
References:
[1] Team, Keras. “Keras Documentation: The Sequential Model.” 
Keras, keras.io/guides/sequential_model. Accessed 13 Aug. 2021.

[2] “Convolutional Neural Networks in Python.” 
DataCamp Community, www.datacamp.com/community/tutorials/convolutional-neural-networks-python. Accessed 13 Aug. 2021

[3] Mittal, Aditya. “Haar Cascades, Explained - Analytics Vidhya.” 
Medium, 26 June 2021, medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d.

[4] GeeksforGeeks. “Keras.Fit() and Keras.Fit_generator().” 
GeeksforGeeks, 25 June 2020, www.geeksforgeeks.org/keras-fit-and-keras-fit_generator.

[5] Rosebrock, Adrian. “How to Use Keras Fit and Fit_generator (a Hands-on Tutorial).” 
PyImageSearch, 5 July 2021, www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial.
'''