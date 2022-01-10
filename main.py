import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

capO=cv2.VideoCapture(0)
capF=cv2.VideoCapture(0)

DEBUG_OUTPUT=True
plt.figure()



def pltShow(title, img):
    if(DEBUG_OUTPUT):
        plt.imshow(img)
        plt.title(title)
        plt.show()

def getBallLoc(region):

    imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)
    blueRegion = cv2.inRange(imgHSV, np.array([105, 200, 0]), np.array([115, 255, 255]))
    pltShow("hsv:", imgHSV)
    pltShow("blueRegion", blueRegion)

    region=cv2.erode(region,np.ones((9,9),np.uint8))
    pltShow('eroded',region)
    region=cv2.dilate(region,np.ones((60,60),np.uint8),3)
    pltShow('dilated',region)
    region=cv2.GaussianBlur(region, (3, 3),0,0)
    regionEdge=cv2.Canny(region,100,300,3)
    img,cnts,hierarchy=cv2.findContours(regionEdge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    pltShow("regionEdge", regionEdge)
    print('轮廓数:',len(cnts))
    cnts=sorted(cnts,key=lambda x:cv2.contourArea(x),reverse=True)

    for i in cnts:
        area = cv2.contourArea(i)
        print('面积:',area)

    cv2.drawContours(regionEdge,cnts,0,(255, 0, 0),thickness=-1)
    pltShow("contour",regionEdge)
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(cnts[0])
    cv2.circle(imgRGB, (cX, cY), 20, color=(255, 255, 0), thickness=-1)
    cv2.rectangle(imgRGB, (x, y), (x + w , y + h), color=(255, 0, 0), thickness=4)
    pltShow('Ball', imgRGB)
    return x,y,w,h,cX,cY

while True:
    sucess, imgRGB = capO.read()
    imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    print(getBallLoc(imgRGB))
    sucess, imgRGB = capF.read()
    imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
    print(getBallLoc(imgRGB))
    time.sleep(5)