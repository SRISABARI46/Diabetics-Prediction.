#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
from twilio.rest import Client


# In[4]:
account_sid = 'AC217c64f6eaae7e4ce831a3941e416104'
auth_token = 'eb46659f516df9c6b2ec765945d856cd'
client = Client(account_sid, auth_token)



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = load_model(r'C:\Users\ASUS\Desktop\Drowsy Detection\model.h5')


# In[10]:


mixer.init()
sound=mixer.Sound(r'C:\Users\ASUS\Desktop\Drowsy Detection\alarm.wav')
cap = cv2.VideoCapture(0)
Score = 0
while True:
    ret, frame = cap.read()
    height,width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
    eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=3)
    
    cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
    for (ex,ey,ew,eh) in eyes:
        #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
        
        eye= frame[ey:ey+eh,ex:ex+ew]
        eye= cv2.resize(eye,(80,80))
        eye= eye/255
        eye= eye.reshape(80,80,3)
        eye= np.expand_dims(eye,axis=0)
        # preprocessing is done now model prediction
        prediction = model.predict(eye)
        # if eyes are closed
        if prediction[0][0]>0.30:
            cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score=Score+1
            if(Score>4):
                try:
                    sound.play()
                    message = client.messages.create(
                        body="Drowsiness Detected! Please wake up!",
                        from_="+19388000892",
                        to="+918838665867"
                    )
                    print("SMS sent successfully!")
                except Exception as e:
                    print("Failed to send SMS:", str(e))
                except:
                    pass
                
        elif prediction[0][1]>0.90:
            cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)      
            cv2.putText(frame,'Score'+str(Score),(100,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                       thickness=1,lineType=cv2.LINE_AA)
            Score = Score-1
            if (Score<0):
                Score=0
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(33) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:

#TWILLIO PHONE NUMBER
    #+15169904452
#Account SID
    #AC217c64f6eaae7e4ce831a3941e416104
#Auth Token
    #eb46659f516df9c6b2ec765945d856cd



#TWILLIO PHONE NUMBER
    #+19388000892
#Account SID
    #AC336ed2ca827625a3d20c8c0a2695cd63
#Auth Token
    #e7ce0d0e08d0456c7925377ccdee1d4d




# In[ ]:




