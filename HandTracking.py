import cv2.cv2 as cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                #print(id,lm)
                height, width, channels = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                #print(id, cx,cy)
                if id == 7:
                    cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLandmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img,"FPS:" + str(int(fps)),(30,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("framename",img)
    cv2.waitKey(1)
