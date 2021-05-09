import cv2
import numpy as np


#image = cv2.imread('data/3.jpg',-1)

video = cv2.VideoCapture(0)

while True:
    _, image = video.read()
    
    paper = cv2.resize(image,(500,500))
    ret, thresh_gray = cv2.threshold(cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY),
                            30, 255, cv2.THRESH_BINARY)
    contours, hie = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for c in contours[:10]:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a green 'nghien' rectangle
        cv2.drawContours(paper, [box], 0, (0, 255, 0),5)

    cv2.imshow('paper', paper)
    #cv2.imwrite('paper.jpg',paper)
    if cv2.waitKey(1) % 0xff == 27:
        break

video.release()
cv2.destroyAllWindows()
