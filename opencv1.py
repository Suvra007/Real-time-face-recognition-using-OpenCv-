from sklearn import svm
import cv2
import numpy as np
from PIL import Image
import os
def train_classifier(data_dir):
    path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces=[]
    ids=[]
    for image in path:
        img=Image.open(image).convert("L")
        imageNp=np.array(img,"uint8")
        id=int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids=np.array(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    
    clf.train(faces,ids)
    clf.write("classifier.yml")

train_classifier("data")

#def generate_dataset(img,id,img_id):
    #cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg",img)
def draw_boundary(img,classifier,scalefactor,minNeighbors,color,text,clf):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(gray_img,scalefactor,minNeighbors)
    coords=[]
    for(x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        id,_=clf.predict(gray_img[y:y+h,x:x+h])
        if id==1:
            cv2.putText(img,"Suvro",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        else:
            cv2.putText(img,"none",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x,y,w,h]
    return coords
def recognize(img,clf,faceCascade):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords=draw_boundary(img,faceCascade,1.1,10,color["white"],"Face",clf)
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
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")


video_capture=cv2.VideoCapture(0)
while True:
    _,img=video_capture.read()
    img=recognize(img,clf,faceCascade)
    cv2.imshow("face detection",img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()

                               
