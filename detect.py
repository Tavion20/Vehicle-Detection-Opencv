import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

model=YOLO('yolov8s.pt')
cap=cv2.VideoCapture("Assets/cars_.mp4")
algo=cv2.createBackgroundSubtractorMOG2(history=10,varThreshold=50)
offset=5
counter=0
fwidth=1200
fheight=700
countline=int(fheight-(fheight/8))
carh=(fheight/500)*80
carw=(fwidth/20)
truckh=carh+(fheight/12)

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
myfile=open("Assets/classes.txt","r")
data=myfile.read()
class_list=data.split("\n")
while True:
    ret,frame=cap.read()
    new_frame=cv2.resize(frame,(fwidth,fheight))
    # gray=cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
    cars=model.predict(new_frame)
    a=cars[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if str(c)=='car':
            color_vehicle=(255,255,0)
        elif str(c)=='truck':
            color_vehicle=(100,100,150)
        elif str(c)=='motorcycle':
            color_vehicle=(255,255,255)
        elif str(c)=='bus':
            color_vehicle=(255,0,255)
        elif str(c)=='bicycle':
            color_vehicle=(0,255,255)
        else:
            continue
        cv2.rectangle(new_frame,(x1,y1),(x2,y2),color_vehicle,2)
        cv2.putText(new_frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        
    # blur=cv2.GaussianBlur(gray,(1,1),7)
    # img_sub=algo.apply(blur)
    # dilate=cv2.dilate(img_sub,np.ones((5,5)))
    # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # dilate_new=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    # morph=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    # counterShape,h=cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(new_frame,(x,y),(x+w,y+h),(0,0,255),2)
    #     cv2.imshow("Original",new_frame)

    # cv2.line(new_frame,(25,countline),((fwidth-25),countline),(0,255,0),3)

    # for (i,c) in enumerate(counterShape):
    #     (x,y,w,h)=cv2.boundingRect(c)
    #     val_count=(w>=carw) and (h>=carh)
    #     if not val_count:
    #         continue
    #     if (h>=truckh):
    #         color_vehicle=(255,255,0)
    #     elif (w<=80):
    #         color_vehicle=(100,100,150)
    #     else:
    #         color_vehicle=(0,0,255)
    #     cv2.rectangle(new_frame,(x,y),(x+w,y+h),color_vehicle,2)
    #     center=center_handle(x,y,w,h)
    #     detect.append(center)
    #     cv2.circle(new_frame,center,4,(255,0,0),2)

    #     for (x,y) in detect:
    #         if y<(countline+offset) and y>(countline-offset):
    #             detect.remove((x,y))
    #             counter+=1
    
    # cv2.putText(new_frame,"VEHICLE COUNT: "+str(counter),(50,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),3)

    cv2.imshow("Original",new_frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cv2.destroyAllWindows()
cap.release()