import requests
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import time
import imutils
from playsound import playsound
import datetime as dt
import threading
from pygame import mixer



#We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()
# We add 'index_camera' argument using add_argument() including a help.
parser.add_argument("index_camera", help="index of the camera to read from", type=int)
args = parser.parse_args()

#Pretrained models and openCV Facecascader
faceCascade = cv2.CascadeClassifier('models/haarcascade-frontalface-default.xml')
emotion_model = load_model('models/_mini_XCEPTION.102-0.66.hdf5', compile=False)
gender_model = load_model('models/gender_mini_XCEPTION.21-0.95.hdf5', compile=False)


#Gender and Emotion arrays
gender=["weiblich","maennlich"]
emotion = ["sauer" ,"","angst", "froehlich", "traurig","","neutral"]


capture=cv2.VideoCapture(args.index_camera)

#Check if camera connected succesfully
if capture.isOpened() is False:
    print(" Error on opening camera")


#Method to download the required models
def downloadFile(url):
    print("Downloading Files")
    url = url
    r = requests.get(url, stream = True)
    folder = "./"
    localFilename = folder + url.split('/')[-1] 
    with open(localFilename, 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024):
            if chunk:
                f.write(chunk)

#Playing Song with Python threading as playsound library doesnt support threading->not implemented yet
def playSound(emotionValue):
    currentTime=dt.datetime.now()
    lastTime=dt.datetime.now()

    if emotionValue>0.5  and ((currentTime-lastTime).seconds>10):
        #time.sleep(10)
        lastTime=dt.datetime.now()
        threading.Thread(target=playsound("happy.mp3"))
    else :
        lastTime=dt.datetime.now()
        threading.Thread(target=playsound("sad.mp3"))  
        


while True:

    ret,frame=capture.read()
    if ret is not None:

        #reduce the noise and change to gray
        frame=imutils.resize(frame, width=650)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        gray=cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5)
        emotion_probability=0
        val=False
        mixer.init()
        
        
        if len(faces)>0:
            currentTime=dt.datetime.now()
            lastTime=dt.datetime.now()

            if val==False and (currentTime-lastTime).seconds>10:
                lastTime=dt.datetime.now()
                mixer.music.load('happy.mp3')
                mixer.music.play()
                
            if val==True  and (currentTime-lastTime).seconds>10:
                lastTime=dt.datetime.now()
                mixer.music.load('sad.mp3')
                mixer.music.play()
                 
                
        
        #loop through faces variable to get the specific values
        for (x, y, w, h) in faces:

            #get variable of the face/s
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            #get the maximum value of emotion prediction
            preds = emotion_model.predict(roi)[0]
            emotion_probability = np.max(preds)
            currentTime=dt.datetime.now();
            #get maximum value of gender prediciton
            preds1=gender_model.predict(roi)[0]
            gender_probability=np.max(preds1)

            print("Emotion:" +str(emotion_probability))
            if emotion_probability>0.5:
                val=True
            if emotion_probability<0.5:
                val=False

            print("Emotion:" +str(emotion_probability)+ str(val))

            #create labels with the results written in-> round to two decimals and cast it to string
            anzeige =[" Emotion: "+str(emotion[preds.argmax()])+" (%"+str(round(emotion_probability*100,2))+") ",
                    " Geschlecht: "+str(gender[preds1.argmax()])+" (%"+str(round(gender_probability*100,2))+") "]

            #create rectangle around the face and also put text above the rectangle
            cv2.rectangle(frame, (x, y-90), (x + w, y -15),(0,255,0),5)
            cv2.rectangle(frame, (x, y-90), (x + w, y -15),(0,255,0),-1)

            rot=[62,30]
            for i in range(len(anzeige)):
                cv2.putText(frame, anzeige[i], (x,y-rot[i]),cv2.FONT_ITALIC, float((w)/600), (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0),3)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key== ord("q"):
            break    
capture.release()
cv2.destroyAllWindows() 