# Emotion-Detection-System
The characteristic of my project is to encourage a strong design that can perceive comparably as see human emotions from a livefeed.

## Introduction
This project targets a person's emotion from a webcamera using Convolutional Neural Networks (CNN) and Haar Cascade Classifiers. The motivation behind this study is to extract main features of emotion detection and its classifiers on the most proficient method to detect the feelings utilizing Convolutional Neural Networks (CNN) and Haar Cascade Classifiers.

![Emotion Detection Block Diagram](https://user-images.githubusercontent.com/82860064/129214953-82147df2-c71c-4b72-9169-7382be9fb573.PNG)

## Dataset
I have used ICML (International Conference on Machine Learning) FERC-2013 dataset. The dataset which we used consists of 3 columns. Emotions, pixels and Usage respectively. Emotion index contains emotion index (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
Pixels will contain 48 x 48 image of grayscale intensity of different emotions which is separated by space. It contains 35,887 grayscale images.
Finally, the usage column will tell whether it is a training dataset or testing dataset.

Download the Dataset from Kaggle: https://www.kaggle.com/msambare/fer2013/download.
Create a folder `data` and extract the contents of the archive into that data folder

## Required packages
OpenCV, Tensorflow.keras, Matplotlib, Numpy, os, Python 3

## Optional packages
If you don't have higher gpu, I would recommend using `Google Colab` and only if you are training your model in Google Colab you would require `from google.colab.patches import cv2_imshow` because cv2.imshow will not work on google colab. After importing the required packages you will need to use cv2_imshow directly to open the webcam.

## Data Training
The `FERC- 2013` dataset is located in the model folder. To train the data you can just run Emotion_trainer.py and the program will run upto 20 epochs. You may change the number of epochs to adjust accuracy. After training the model a model file will be saved naming `model.h5`.



## Data Loading and Predictions
To predict the emotions run Load_model.py. As soon as you load the model, it will start the webcam and starts predicting your emotions. The accuracy is 65 % in 20 epochs. It is a `4-layer Convolutional Neural Network`. To draw the box around the face, you will need to Haar-Cascade. I have included `haarcascade_frontalface_default.xml` file that is used in Load_model.py

## Results
Overall Accuracy achieved: 65 %
### Manual Testing Accuracy Table:
![Manual Testing Table](https://user-images.githubusercontent.com/82860064/129216619-7228455c-723e-4df1-87e9-056bb2469c41.PNG)


## Acknowledgements
I would like to thank the authors from different associations and conferences that helped me contribute to develop this project. 

## References
* Jaiswal, Akriti, A. Krishnama Raju, and Suman Deb. "Facial emotion detection using deep learning." 2020 International Conference for Emerging Technology (INCET). IEEE, 2020.
* Sang, Dinh Viet, and Nguyen Van Dat. "Facial expression recognition using deep convolutional neural networks." 2017 9th International Conference on Knowledge and Systems   Engineering (KSE). IEEE, 2017.
* Mehendale, Ninad. "Facial emotion recognition using convolutional neural networks (FERC)." SN Applied Sciences 2.3 (2020): 1-8.
* Zhou, Shuai, et al. "Facial expression recognition based on multi-scale cnns." Chinese Conference on Biometric Recognition. Springer, Cham, 2016.
*  Puri, Raghav, et al. "Emotion detection using image processing in python." arXiv preprint arXiv:2012.00659 (2020).


