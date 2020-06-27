import cv2
import numpy as np
from skimage import feature 
from PIL import Image
import os
from sklearn.neighbors import KNeighborsClassifier
def describe(image,p,r):
    lbp=feature.local_binary_pattern(image,p,r,method="uniform")
    hist,_= np.histogram(lbp.ravel(),bins=np.arange(0,p+3),range=(0,p+2))
    hist=hist.astype("float")
    hist/=(hist.sum()+1e-7)
    return hist


path=[os.path.join("data",f) for f in os.listdir("data")]
data=[]
labels=[]
for image in path:
        img=Image.open(image).convert("L")
        hist=describe(img,24,8)
        data.append(hist)
        id=int(os.path.split(image)[1].split(".")[1])
        labels.append(id)
model=KNeighborsClassifier(n_neighbors=5,weights="uniform",algorithm="auto",metric="minkowski")
model.fit(data,labels)

def draw_boundary(img,classifier,scalefactor,minNeighbors,color,text,model):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(gray_img,scalefactor,minNeighbors)
    coords=[]
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        hist=describe(gray_img[y:y+h,x:x+h],24,8)
        hist.astype("float")
        id=model.predict(hist.reshape(1,-1))
        if id==1:
            cv2.putText(img,"Suvro",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        elif id==2:
            cv2.putText(img,"Indrani",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        else:
            cv2.putText(img,"None",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x,y,w,h]
    return coords
def recognize(img,model,faceCascade):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords=draw_boundary(img,faceCascade,1.1,10,color["white"],"Face",model)
    return img

def detect(img,faceCascade,eyeCascade,img_id):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords=draw_boundary(img,faceCascade,1.1,10,color["blue"],"Face")

    if len(coords)==4:
        
        roi_img=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        #user_id=1
        #generate_dataset(roi_img,user_id,img_id)

        
        coords=draw_boundary(roi_img,eyeCascade,1.1,18,color["red"],"Eye")
        
    return img



faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
img_id=0


video_capture=cv2.VideoCapture(0)
while True:
    _,img=video_capture.read()
    img=recognize(img,model,faceCascade)
    cv2.imshow("face detection",img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()

