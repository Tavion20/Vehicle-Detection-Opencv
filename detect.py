import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from tracker import*

model = YOLO('yolov8s.pt')

tracker = Tracker()

capture = cv2.VideoCapture("Assets/veh2.mp4")
capture.set(cv2.CAP_PROP_BUFFERSIZE,2) #supposedly increases the fps but doesn't work for now
# algo = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=50) //ahh it's for shadow detection and stuff, no need really

offset = 5
count = 0
fwidth = 1200
fheight = 700
# countline = int(fheight-(fheight/8))
# carh = (fheight/500)*80
# carw = (fwidth/20)
# truckh = carh+(fheight/12)

l1y=int(450.8) #line1 y co-ordinates
l2y= int(515.2) #line 2 y-co-ordinates


vh_down ={}
vh_up ={}
counter_down=[]
counter_up=[]

'''
for getting the positions for lines

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

'''

# def center_handle(x, y, w, h):
#     x1 = int(w/2)
#     y1 = int(h/2)
#     cx = x+x1
#     cy = y+y1
#     return cx, cy


myfile = open("Assets/classes.txt", "r")
data = myfile.read()
class_list = data.split("\n")

while True:
    ret, frame_ = capture.read()
    if not ret:
        break
    # count+=1
    # if count % 3 != 0:
    #     continue
    # frame = algo.apply(frame_)
    new_frame = cv2.resize(frame_, (fwidth, fheight))

    # gray=cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
    results = model.predict(new_frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    detect = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        if str(c) == 'car':
            color_vehicle = (255, 255, 0)
            detect.append([x1,y1,x2,y2])

        # elif str(c) == 'truck':
        #     color_vehicle = (100, 100, 150)

        # elif str(c) == 'motorcycle':
        #     color_vehicle = (255, 255, 255)

        # elif str(c) == 'bus':
        #     color_vehicle = (255, 0, 255)

        # elif str(c) == 'bicycle':
        #     color_vehicle = (0, 255, 255)

        # else:
        #     continue

        cv2.rectangle(new_frame, (x1, y1), (x2, y2), color_vehicle, 2)
        cv2.putText(new_frame, str(c), (x1, y1+3),cv2.FONT_HERSHEY_COMPLEX, 0.5, (57, 255, 20), 2)

    # blur=cv2.GaussianBlur(gray,(1,1),7)
    # img_sub=algo.apply(blur)
    # dilate=cv2.dilate(img_sub,np.ones((5,5)))
    # kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # dilate_new=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    # morph=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    # counterShape,h=cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for (x,y,w,h) in results:
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


    bbox_id = tracker.update(detect)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        #going down
        if l1y < (cy + offset) and l1y > (cy - offset):
            vh_down[id] = cy
        if id in vh_down:
            if l2y < (cy + offset) and l2y > (cy - offset):
                cv2.circle(new_frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(new_frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                counter_down.append(id)

        #going up
        if l2y < (cy + offset) and l2y > (cy - offset):
            vh_up[id] = cy
        if id in vh_up:
            if l1y < (cy + offset) and l1y > (cy - offset):
                cv2.circle(new_frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(new_frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                counter_up.append(id)
           
    #this is for video -> cars_.mp4
    cv2.line(new_frame,(314,l1y),(975,l1y),(255,255,255),2)
    cv2.putText(new_frame,"Line 1",(324,l1y-7),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.line(new_frame,(196,l2y),(1096,l2y),(255,255,255),2)
    cv2.putText(new_frame,"Line 2",(206,l2y-7),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.putText(new_frame,f"Going down: {len(set(counter_down))}",(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)
    cv2.putText(new_frame,f"Going up: {len(set(counter_up))}",(60,70),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),2)

    ##########

    cv2.imshow("RGB", new_frame)
    if cv2.waitKey(1) & 0xFF==ord("d"): #press key d exits from the video 
        break

capture.release()
cv2.destroyAllWindows()
