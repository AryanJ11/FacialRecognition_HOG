# -*- coding: utf-8 -*-
"""
@author: ARYAN JAIN

"""

import cv2
import numpy as np
from skimage.feature import hog
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils
dim=(64, 128)
face_cas=cv2.CascadeClassifier(r'C:\Users\ARYAN JAIN\Desktop\haarcascade_frontalface_alt2.xml')

#STEP1: INPUT THE DATA. 
# The data must include atleast 10 images each of the faces to be recognized and classified
# as well as images which do not contain any of the said faces. I have also used Viola-Jones
# method to extract faces form the raw images used and these faces are then resized to dimensions of 64x128.
# This size is optimal for HOGs. Moreover the size of all images in the dataset must be the same so that the
# length of the feature vectors of all images is equal

#This is a binary classification. However more relevant changes can be made for multi=classififcation
#image:1-9 are of the faces to be recognized
imgg1=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0018.jpg')
gray1=cv2.cvtColor(imgg1, cv2.COLOR_BGR2GRAY)
face1=face_cas.detectMultiScale(gray1, 1.3, 5)
for (x,y,w,h) in face1:
   roi_color1 = imgg1[y:y+h, x:x+w]
img1=cv2.resize(roi_color1, dim)

imgg2=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0019.jpg')
gray=cv2.cvtColor(imgg2, cv2.COLOR_BGR2GRAY)
face=face_cas.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in face:
   roi_color = imgg2[y:y+h, x:x+w]
img2=cv2.resize(roi_color, dim)

imgg3=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0020.jpg')
gray3=cv2.cvtColor(imgg3, cv2.COLOR_BGR2GRAY)
face3=face_cas.detectMultiScale(gray3, 1.3, 5)
for (x,y,w,h) in face3:
   roi_color3 = imgg3[y:y+h, x:x+w]
img3=cv2.resize(roi_color3, dim)

imgg4=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0021.jpg')
gray4=cv2.cvtColor(imgg4, cv2.COLOR_BGR2GRAY)
face4=face_cas.detectMultiScale(gray4, 1.3, 5)
for (x,y,w,h) in face4:
   roi_color4 = imgg4[y:y+h, x:x+w]   
img4=cv2.resize(roi_color4, dim)
  
imgg5=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0027.jpg')
gray5=cv2.cvtColor(imgg5, cv2.COLOR_BGR2GRAY)
face5=face_cas.detectMultiScale(gray5, 1.3, 5)
for (x,y,w,h) in face5:
   roi_color5 = imgg5[y:y+h, x:x+w]
img5=cv2.resize(roi_color5, dim)

imgg6=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0023.jpg')
gray6=cv2.cvtColor(imgg6, cv2.COLOR_BGR2GRAY)
face6=face_cas.detectMultiScale(gray5, 1.3, 5)
for (x,y,w,h) in face6:
   roi_color6 = imgg6[y:y+h, x:x+w]
img6=cv2.resize(roi_color6, dim)

imgg7=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0024.jpg')
gray7=cv2.cvtColor(imgg7, cv2.COLOR_BGR2GRAY)
face7=face_cas.detectMultiScale(gray7, 1.3, 5)
for (x,y,w,h) in face7:
   roi_color7 = imgg7[y:y+h, x:x+w]
img7=cv2.resize(roi_color7, dim)

imgg8=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0025.jpg')
gray8=cv2.cvtColor(imgg8, cv2.COLOR_BGR2GRAY)
face8=face_cas.detectMultiScale(gray5, 1.3, 5)
for (x,y,w,h) in face8:
   roi_color8 = imgg8[y:y+h, x:x+w]
img8=cv2.resize(roi_color8, dim)

imgg9=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\IMG-20200327-WA0026.jpg')
gray9=cv2.cvtColor(imgg9, cv2.COLOR_BGR2GRAY)
face9=face_cas.detectMultiScale(gray9, 1.3, 5)
for (x,y,w,h) in face9:
   roi_color9 = imgg9[y:y+h, x:x+w]
img9=cv2.resize(roi_color9, dim)

#image:10-17 are of arbitary faces.

imgg10=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk1.jpg')
gray10=cv2.cvtColor(imgg10, cv2.COLOR_BGR2GRAY)
face10=face_cas.detectMultiScale(gray9, 1.3, 5)
for (x,y,w,h) in face10:
   roi_color10 = imgg10[y:y+h, x:x+w]
img10=cv2.resize(roi_color10, dim)
imgg11=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk2.jpg')
gray11=cv2.cvtColor(imgg11, cv2.COLOR_BGR2GRAY)
face11=face_cas.detectMultiScale(gray11, 1.3, 5)
for (x,y,w,h) in face11:
   roi_color11 = imgg11[y:y+h, x:x+w]
img11=cv2.resize(roi_color11, dim)
imgg12=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk3.jpg')
gray12=cv2.cvtColor(imgg12, cv2.COLOR_BGR2GRAY)
face12=face_cas.detectMultiScale(gray12, 1.3, 5)
for (x,y,w,h) in face12:
   roi_color12 = imgg12[y:y+h, x:x+w]
img12=cv2.resize(roi_color12, dim)
imgg13=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk4.jpg')   
gray13=cv2.cvtColor(imgg13, cv2.COLOR_BGR2GRAY)
face13=face_cas.detectMultiScale(gray13, 1.3, 5)
for (x,y,w,h) in face13:
   roi_color13 = imgg13[y:y+h, x:x+w]
img13=cv2.resize(roi_color13, dim)
imgg14=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk5.jpg')
gray14=cv2.cvtColor(imgg14, cv2.COLOR_BGR2GRAY)
face14=face_cas.detectMultiScale(gray14, 1.3, 5)
for (x,y,w,h) in face14:
   roi_color14 = imgg14[y:y+h, x:x+w]
img14=cv2.resize(roi_color14, dim)
imgg15=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk6.jpg')
gray15=cv2.cvtColor(imgg15, cv2.COLOR_BGR2GRAY)
face15=face_cas.detectMultiScale(gray15, 1.3, 5)
for (x,y,w,h) in face15:
   #cv2.rectangle(test,(x,y),(x+w,y+h),(255,255,0),5)
   roi_color15 = imgg15[y:y+h, x:x+w]
img15=cv2.resize(roi_color15, dim)
imgg16=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk7.jpg')   
gray16=cv2.cvtColor(imgg16, cv2.COLOR_BGR2GRAY)
face16=face_cas.detectMultiScale(gray16, 1.3, 5)
for (x,y,w,h) in face16:
   roi_color16 = imgg16[y:y+h, x:x+w]
img16=cv2.resize(roi_color16, dim)
imgg17=cv2.imread(r'C:\Users\ARYAN JAIN\Desktop\web-d\uk8.jpg')
gray17=cv2.cvtColor(imgg17, cv2.COLOR_BGR2GRAY)
face17=face_cas.detectMultiScale(gray17, 1.3, 5)
for (x,y,w,h) in face17:
   roi_color17 = imgg17[y:y+h, x:x+w]
img17=cv2.resize(roi_color17, dim)

#STEP2: EXTRACT THE FEATURE VECTORS. 
# 'hog' is an in-built function that calculates the feature vector of images that are provided. For further info
# check its documentation. Since size of images in dataset is 64x128 the size of feature vectors of the images will be
# 3780 each. The feature vectors are then arranged in a row which will then be fed into the classifier.  

fd1, him=hog(img1, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd2, him=hog(img2, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd3, him=hog(img3, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd4, him=hog(img4, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd5, him=hog(img5, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd6, him=hog(img6, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd7, him=hog(img7, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd8, him=hog(img8, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)
fd9, him=hog(img9, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True)                     
fd10, him=hog(img10, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd11, him=hog(img11, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd12, him=hog(img12, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd13, him=hog(img13, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd14, him=hog(img14, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd15, him=hog(img15, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd16, him=hog(img16, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
fd17, him=hog(img17, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
X=np.row_stack([fd1, fd2, fd3, fd4, fd5, fd6, fd7, fd8, fd9, fd10, fd11, fd12, fd13, fd14, fd15, fd16, fd17])
# A better implementation would be to run a loop throughout the data since same task is performed on every image....

#STEP3: Declaring and training a SVM classifier.
# Linear svm can also be used. However in my case it did not provide a good classification and training it was also harder. Once every image is
# given a name it has to be converted to corresponding numbers so that it can be used by svm. Label Encoder coverts each name to a number denoting
# the class to which it belongs. After this the svm is trained. To provide the best fit several options of C and Gamma can be tested at the same time.     
name=["Welcome","Welcome","Welcome","Welcome","Welcome","Welcome","Welcome","Welcome","Welcome","Unknown","Unknown","Unknown","Unknown","Unknown","Unknown","Unknown","Unknown"]
le=preprocessing.LabelEncoder()
Y=le.fit_transform(name)
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1] }
recog=GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid)
recog=recog.fit(X,Y)
#print(recog.best_estimator_)  -- can be used to check the paramters used by the svm

#STEP4: RECOGNIZING FACES IN VIDEO-STREAMS.
# Faces ar extracted form each frame and then the svm works on that to classify it as known or unknown. The class with maximum probability
# is chosen with some other conditions posed on it as well.
#NOTE: Label Encoder labels 'Welcome' as class:0 while 'Unknown' as class:1. However when the classes are called using le.classes_ the
# names appear in alphabetical order. Hence 'Unknoen; is at index:0 while 'Welcome' at index:1. Therefore j is changed appropriately to 
# display the correct name on the screen. Morever to reduce false positives only those faces which are classified as 0 with a probability 
# of more than 60% are identified as 'Welcome'.

cap=cv2.VideoCapture(0)
time.sleep(2.0)
#start the FPS throughout estimator
fps = FPS().start()
while(True):
    ret,test=cap.read()
    test= imutils.resize(test, width=600)
    gray=cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    face_cas=cv2.CascadeClassifier(r'C:\Users\ARYAN JAIN\Desktop\haarcascade_frontalface_alt2.xml')
    face=face_cas.detectMultiScale(gray, 1.3, 5)
    if len(face)!=0:
      for (x,y,w,h) in face:
        roi_color = test[y:y+h, x:x+w]
        testt=cv2.resize(roi_color,dim)
        fdt, _=hog(testt, orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True,multichannel=True)
        fdd=np.row_stack([fdt])
        pred=recog.predict_proba(fdd)
        print(pred)
        jj=np.amax(pred)
        j=np.argmax(pred)
        proba=jj
        if j==0 and proba>0.6:
            j=1
            cv2.putText(test,"Access granted",(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)
        elif j==0 and proba<=0.6:
            j=0
            proba=1-jj
            cv2.putText(test,"Access denied",(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        elif j==1:
            j=0
            cv2.putText(test,"Access denied",(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),3)
        names=le.classes_[j]
        text="{}: {:.2f}%".format(names, proba * 100)
        for (x,y,w,h) in face:
              cv2.rectangle(test,(x,y),(x+w,y+h),(255,255,0),5)
              cv2.putText(test, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)      
    cv2.imshow("vid",test)      
    if cv2.waitKey(1)==ord('q') & 0xff:
        break
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))   
cap.release()
cv2.destroyAllWindows()