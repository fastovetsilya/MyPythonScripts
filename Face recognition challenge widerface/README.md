## A comparison between Haar Cascade Classifier and Multi-task Cascaded Convolutional Neural Networks (MTCNN) for face detection task

In this project I compare two models for face detection on WIDERFACE dataset: http://shuoyang1213.me/WIDERFACE/. 

### Haar Cascade Classifier

At the first step I try the CascadeClassifier class from OpenCV module to detect faces and estimate its precision and recall.
The algorithm is based on AdaBoost classifier trained on Haar features of the faces. The complete description of the method may be found on the OpenCV website: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html

### MTCNN

At the second step I run Multitask Cascaded Convolutional Networks (MTCNN) to detect faces. An MTCNN is an architecture that combines three neural networks to suggest the best bbx for faces. Each of the three models are trained separately. For this reason, it is very difficult to train from scratch, but pre-trained models are available. For this project we use the following implementation of MTCNN: https://github.com/jbrownlee/mtcnn . MTCNN was developed in 2016 [https://arxiv.org/abs/1604.02878] and today is considered one of the best-performing models. 

### Comparing model performance

It is shown that MTCNN model has much better performance in terms od IoU-constrained precision and recall on a face detection task, compared to Haar Cascade Classifyer with the same IoU detection threshold. The difference between the methods is the biggest in terms of precision: the recognised objects are much more likely to be faces when MTCNN model is used. This example demonstrates that the approach based on Convolutional Neural Networks is capable of detecting features in the images much better compared to other techniques. A bad recall in both models may be due to the fact that most faces in the images were difficult to detect for a variety of reasons. 