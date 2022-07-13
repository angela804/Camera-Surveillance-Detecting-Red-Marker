import cv2
import time
import numpy as np


TIMER = int(3)

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    cv2.imshow("a",img)

    k = cv2.waitKey(125)

    if TIMER == 3:
        prev = time.time()
        while TIMER >= 0:
            ret, img = cap.read()
            cv2.imshow('a', img)
            cv2.waitKey(125)
            cur = time.time()
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1 
        else:
            ret, img = cap.read()
            cv2.imshow('a', img)
            cv2.waitKey(2000)
            cv2.imwrite('camera.jpg', img)
            path = r"C:\Users\Admin\Desktop\Researcher\camera.jpg"

            image = cv2.imread(path)
            img = cv2.resize(image,(512,383))

            img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # lower mask (0-10)
            lower_red = np.array([0,100,100]) # 0,50,50 original
            upper_red = np.array([10,255,255]) 
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red) 

            # upper mask (170-180)
            lower_red = np.array([160,100,100]) # 170,50,50 original
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            mask = mask0+mask1

            output_img = img.copy()
            output_img[np.where(mask==0)] = 0
            output_img[np.where(mask!=0)] = 255

            pixels = np.where(mask==[0]) #Pixels for the Undetected
            pixels2 =np.where(mask==[255]) #Pixels for the Detected

            circle_centerX = int((max(pixels2[1]) + min(pixels2[1])) / 2) #Getting the Center of the Marker
            circle_centerY = int((max(pixels2[0]) + min(pixels2[0])) / 2)

            image_centerX = int((max(pixels[1]) + min(pixels[1])) / 2) #Getting the Center of the Image
            image_centerY = int((max(pixels[0]) + min(pixels[0])) / 2) #The center of the image, by sense, marks the location of the UGV based on the picture

            x_distance = ((circle_centerX-image_centerX)**2 + (circle_centerY-image_centerY)**2)**(1/2) #Euclidean Distance between the UGV and the Red Marker based on a 2D space (x-coordinates)

            print(len(pixels[0]))
            print(len(pixels2[0]))
            print("Distance between the UGV and the Red Marker: ", x_distance)

    TIMER = int(3)

    if k == 27:
        break


cap.release()

cv2.destroyAllWindows()
