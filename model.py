import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 50)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
left = False
right = False

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listimg = os.listdir("imgs")
print(listimg)
imgList = []
for imgPath in listimg:
    img = cv2.imread(f"imgs/{imgPath}")
    rsimgBg = cv2.resize(img, (640, 480))
    imgList.append(rsimgBg)
print(len(imgList))
indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.4)

    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    _, imgStacked = fpsReader.update(imgStacked)
    print(indexImg)
    cv2.imshow("Image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    # if key == ord("a"):
    #      if indexImg > 0:
    #         indexImg -= 1
    # elif key == ord("d"):
    #     if indexImg < len(imgList) - 1:
    #         indexImg += 1
    # elif key == ord("q"):
    #     break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                # print(idx,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        if not left and handPoints[8][0] < 220:
            left = True
            if indexImg > 0:
                indexImg -= 1
                print("Left")
        if not right and handPoints[8][0] > 420:
            right = True
            if indexImg < len(imgList) - 1:
                indexImg += 1
                print("right")

        if handPoints[8][0] > 220 and handPoints[8][0] < 420:
            left = False
            right = False
