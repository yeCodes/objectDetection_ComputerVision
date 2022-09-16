# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:28:47 2022

@author: olise
"""

import numpy
import cv2 as cv

import os

# necessary to avoid working directory from changing, then key files not being located
# https://stackoverflow.com/questions/38393628/set-working-directory-in-python-spyder-so-that-its-reproducible
def enterworkingDirectoryPath(path):
    os.chdir(path)  # without r, it cannot read spaces

print(cv.__version__)

def runCameraOne():
    # https://stackoverflow.com/questions/604749/how-do-i-access-my-webcam-in-python
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
    while rval:
        cv.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv.waitKey(20)
        if key == 27: # exit on ESC - press escape to ext
            break
    
    vc.release()
    cv.destroyWindow("preview")

def runCameraTwo():
    # https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
    
    # define a video capture object
    vid = cv.VideoCapture(0)
      
    while(True):
          
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
      
        # Display the resulting frame
        cv.imshow('frame', frame)
          
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            "press 'q' to quit"
            break
      
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    
def objectDetectionTest():
    cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
    
    img = cv.imread('example.jpg')
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    
    cv.imshow('img', img)
    cv.waitKey()
    

from matplotlib import pyplot as plt

def openImage():
  
    # https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
    # Opening image
    img = cv.imread("image.jpg")
      
    # OpenCV opens images as BRG 
    # but we want it as RGB and 
    # we also need a grayscale 
    # version
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      
    # Creates the environment 
    # of the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
    
def singleObjectDetectionImgTest(imageInDirectory):
    # https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
    # Opening image
    img = cv.imread(imageInDirectory) #("dogHorse.jpg") #("image.jpg")
      
    # OpenCV opens images as BRG 
    # but we want it as RGB We'll 
    # also need a grayscale version
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      
      
    # Use minSize because for not 
    # bothering with extra-small 
    # dots that would look like STOP signs
    stop_data = cv.CascadeClassifier('stop_data.xml')
      
    found = stop_data.detectMultiScale(img_gray, 
                                       minSize =(20, 20))
      
    # Don't do anything if there's 
    # no sign
    amount_found = len(found)
      
    if amount_found != 0:
          
        # There may be more than one
        # sign in the image
        for (x, y, width, height) in found:
              
            # We draw a green rectangle around
            # every recognized sign
            cv.rectangle(img_rgb, (x, y), 
                          (x + height, y + width), 
                          (0, 255, 0), 5)
              
    # Creates the environment of 
    # the picture and shows it
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.show()
    
def objectDetV2(imageFile):
    
    # first n mins of - https://www.youtube.com/watch?v=RFqvTmEFtOE
    # dnn weights - "...a configuration file which was used for training to help script determine hyper-parameters."
    # 80 classes/ labels in coco dataset
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    # the dnn structure == architecture
    frozen_model_struct = 'frozen_inference_graph.pb'

    # this step - https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API    
    cvNet = cv.dnn.readNetFromTensorflow(frozen_model_struct, config_file)

    img = cv.imread(imageFile)
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    
    cv.imshow('img', img)
    cv.waitKey()    

def initialiseOpencvModel():
    ## the weights of the trained dnn
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    # the dnn structure == architecture
    frozen_model_struct = 'frozen_inference_graph.pb'

    # this step - https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API    
    model = cv.dnn_DetectionModel(frozen_model_struct, config_file)
    
    classLabels = [] 
    file_name = "Labels.txt"  # https://github.com/pjreddie/darknet/blob/master/data/coco.names
    with open(file_name, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
        
    return model, classLabels

def objectDetTutorial(imageInDirectory):
    # required input data type: image
    
    model, classLabels = initialiseOpencvModel()
    print(classLabels)
    
    ## read an image
    img = cv.imread(imageInDirectory)
    
    # setup model configuration
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)      #255/2 = 127.5
    model.setInputMean((127.5,127.5,127.5)) 
    model.setInputSwapRB(True)
    
    # print image to IDE output
    plt.imshow(img)     #bgr image. default using opencv
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))   # convert bgr image to rgb. presumably so readable by most other image readers.
    
    ClassIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)  # accuracy when predicting
    print(ClassIndex)
    
    for i in ClassIndex:
        print(classLabels[i])
        
    # draw rectangle on image + set font
    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN
    
    # flattening NEXTED arrays!!
    for ClassIndex, confidence, boxes in zip(ClassIndex.flatten( ), confidence.flatten(), bbox):
        cv.rectangle(img, boxes,(255,0,0),2)
        cv.putText(img, classLabels[ClassIndex-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color = (0,255,0), thickness = 3)
    
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))   # convert bgr image to rgb. presumably so readable by most other image readers.

def configureDNNInputs(model):
    # setup model configuration - yields error without this. 
    # Need to condition model input so right size
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)      #255/2 = 127.5
    model.setInputMean((127.5,127.5,127.5)) 
    model.setInputSwapRB(True)
    
    return None
    
def videoObjectDet(path):
    
    video = cv.VideoCapture(path)
    
    # check if vid opened correctly - not certain code will work
    if not video.isOpened():
        video = cv.VideoCapture(0)  # run webcam
    if not video.isOpened():
        # raise error
        raise IOError("Cannot open video")
        
    
    model, classLabels = initialiseOpencvModel()
    configureDNNInputs(model)
    runVideoDnnDetection(video, model, classLabels)
    
def runVideoDnnDetection(video, model, classLabels):
    
    # format video text
    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN 
    
    while True:
        # runs until break condition within window met
        ret, frame = video.read()       # read video by frame!!
        
        ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)
        
        #print(ClassIndex)
        # print name of object console. When no object detected in areas, openCV
        # seems to output 84 to the console. len(classLabels)-1 prevents out of bounds error
        for i in ClassIndex:
            if i <= len(classLabels)-1:
                print(classLabels[i-1])
                
        if (len(ClassIndex)!=0):
            for ClassIndex, confidence, boxes in zip(ClassIndex.flatten( ), confidence.flatten(), bbox):
                if(ClassIndex<=80):
                    cv.rectangle(frame, boxes,(255,0,0),2)
                    cv.putText(frame, classLabels[ClassIndex-1], (boxes[0]+10, boxes[1]+40), font, fontScale = font_scale, color = (0,255,0), thickness = 3)
          
        cv.imshow('Object Detection Tutorial', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            "press 'q' to quit"
            break
    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv.destroyAllWindows()
    
    
def webcamObjectDet():
    
    video = cv.VideoCapture(0)  # run webcam. Get stream from video.
        
    model, classLabels = initialiseOpencvModel()
    configureDNNInputs(model)
    runVideoDnnDetection(video, model, classLabels)
    
    
if __name__=="__main__":
    

    path = '************ENTER FULL FILEPATH HERE*****************'
    enterworkingDirectoryPath(path)
    
    
    #singleObjectDetectionImgTest("dogHorse.jpg")
    #singleObjectDetectionImgTest("image.jpg")
    #objectDetV2("restaurant11.jpg")
        
    #objectDetTutorial('testCar.jpg')    # images with complex backgrounds not working as well
    
    #runCameraTwo()
    videoObjectDet("walkingLondon.mp4")
    #webcamObjectDet()