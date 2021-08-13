# Facial Emotion Detection System
The characteristic of my project is to encourage a strong design that can perceive comparably as see human emotions from a livefeed.

## Introduction
This project targets a person's emotion from a webcamera using Convolutional Neural Networks (CNN) and Haar Cascade Classifiers. The motivation behind this study is to extract main features of emotion detection and its classifiers on the most proficient method to detect the feelings utilizing Convolutional Neural Networks (CNN) and Haar Cascade Classifiers.

![Emotions](https://user-images.githubusercontent.com/37297153/129219805-ff818916-db0b-4964-8901-3e9f43f5dc45.png)


## Dataset
I have used ICML (International Conference on Machine Learning) FERC-2013 dataset. The dataset which we used consists of 3 columns. Emotions, pixels and Usage respectively. Emotion index contains emotion index (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
Pixels will contain 48 x 48 image of grayscale intensity of different emotions which is separated by space. It contains 35,887 grayscale images.
Finally, the usage column will tell whether it is a training dataset or testing dataset.

Download the Dataset from Kaggle: https://www.kaggle.com/msambare/fer2013/download.


## Required packages
OpenCV, Tensorflow.keras, Matplotlib, Numpy, os, Python 3

## Optional packages
If you don't have higher gpu, I would recommend using `Google Colab` and only if you are training your model in Google Colab you would require `from google.colab.patches import cv2_imshow` because cv2.imshow will not work on google colab. After importing the required packages you will need to use cv2_imshow directly to open the webcam.

# Usage 
## Data Training
The `FERC- 2013` dataset is located in the model folder. To train the data you can just run Emotion_trainer.py and the program will run upto 20 epochs. You may change the number of epochs to adjust accuracy. After training the model a model file will be saved naming `model.h5`.


## Data Loading and Predictions
To predict the emotions run Load_model.py. As soon as you load the model, it will start the webcam and starts predicting your emotions. The accuracy is 65 % in 20 epochs. It is a `4-layer Convolutional Neural Network`. To draw the box around the face, you will need to Haar-Cascade. I have included `haarcascade_frontalface_default.xml` file that is used in Load_model.py

# Results
Overall Accuracy achieved: 65 %

# Structure
Create a folder `data` and extract the contents of the archive into that data folder. 
## Acknowledgements
I would like to thank the authors from different associations and conferences that helped me contribute to develop this project. 

## References
* Jaiswal, Akriti, A. Krishnama Raju, and Suman Deb. "Facial emotion detection using deep learning." 2020 International Conference for Emerging Technology (INCET). IEEE, 2020.
* Sang, Dinh Viet, and Nguyen Van Dat. "Facial expression recognition using deep convolutional neural networks." 2017 9th International Conference on Knowledge and Systems   Engineering (KSE). IEEE, 2017.
* Mehendale, Ninad. "Facial emotion recognition using convolutional neural networks (FERC)." SN Applied Sciences 2.3 (2020): 1-8.
* Zhou, Shuai, et al. "Facial expression recognition based on multi-scale cnns." Chinese Conference on Biometric Recognition. Springer, Cham, 2016.
* Puri, Raghav, et al. "Emotion detection using image processing in python." arXiv preprint arXiv:2012.00659 (2020).
* Verma, Prakash. “Detect Facial Emotions on Mobile and IoT Devices Using TensorFlow Lite.” Medium, 20 Jan. 2021, heartbeat.fritz.ai/detect-facial-emotions-on-mobile-and-iot- devices-using-tensorflow-lite-e98e7a48c309.
* Team, Keras. “Keras Documentation: The Sequential Model.” Keras, keras.io/guides/sequential_model. Accessed 13 Aug. 2021.
* “Convolutional Neural Networks in Python.” DataCamp Community, www.datacamp.com/community/tutorials/convolutional-neural-networks-python. Accessed 13 Aug. 2021
* Mittal, Aditya. “Haar Cascades, Explained - Analytics Vidhya.” Medium, 26 June 2021, medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d.
* GeeksforGeeks. “Keras.Fit() and Keras.Fit_generator().” GeeksforGeeks, 25 June 2020, www.geeksforgeeks.org/keras-fit-and-keras-fit_generator.
* Rosebrock, Adrian. “How to Use Keras Fit and Fit_generator (a Hands-on Tutorial).” PyImageSearch, 5 July 2021, www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial.

