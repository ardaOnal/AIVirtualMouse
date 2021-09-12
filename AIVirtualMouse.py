import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import math

widthCam, heightCam = 640, 480
widthScreen, heightScreen = autopy.screen.size() # get the height of the screen
ROIsize = 50 # region of interest size
smoothening = 6 # smootheing the movement of mouse factor
previousTime = 0 # for fps calculation
detector = htm.HandDetector(maxHands=1) # hand detector
plocX,plocY = 0,0 # previous location of mouse
clocX,clocY = 0,0 # current location of mouse

cap = cv2.VideoCapture(0)
cap.set(3,widthCam)
cap.set(4,heightCam)

while True:
    success, img = cap.read() # get the frame
    img = cv2.flip(img,1) # flip the image so that when our hand goes to right, mouse moves to right

    # Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img) # get the landmarks and the bounding box
    cv2.rectangle(img, (ROIsize, ROIsize), (widthCam - ROIsize, heightCam - ROIsize*4), (255, 255, 0), 2) # draw the region of interest

    # Get the tip of the fingers
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:] # index finger
        #x2,y2 = lmList[4][1:] # thumb

        # Check which of fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        # Check if it is in moving mode
        if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Convert coordinates
            x3 = np.interp(x1,(ROIsize,widthCam-ROIsize),(0,widthScreen))
            y3 = np.interp(y1,(ROIsize,heightCam-ROIsize*4),(0,heightScreen))

            # Smoothen the values (reduce noise)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            plocX , plocY = clocX, clocY # update previous locations

            # Move mouse
            autopy.mouse.move(clocX,clocY)
            cv2.circle(img,(x1,y1),15,(0,255 ,255),cv2.FILLED) # when mouse is moving, draws circle

        # Check if it is clicking mode
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # Find distance between fingers
            length, img, _ = detector.findDistance(4,8,img)

            # Click if distance is small
            if length < 75:
                cv2.circle(img,(x1,y1),15,(0,255 ,0),cv2.FILLED) # when mouse clicks, draws circle
                autopy.mouse.click() #  with mouse

    # Framerate
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img,"fps" + str(int(fps)),(20,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),3)

    # Display
    cv2.imshow("frame",img)
    cv2.waitKey(1)