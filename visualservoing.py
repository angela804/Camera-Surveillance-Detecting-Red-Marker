import cv2
from matplotlib import image
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = "Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

#path = r'C:\Users\Admin\Pictures\rose.jpg'  #location of the image
#image = cv2.imread(path, 1)
#img = cv2.resize(image,(512,383)) #Dili mag'matter ang resize sa coordinates
img = image

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
output_img[mask==0] = 0
output_img[mask!=0] = 255

pixels = np.where(mask==[0]) #Pixels for the Undetected
pixels2 =np.where(mask==[255]) #Pixels for the Detected

circle_centerX = int((max(pixels2[1]) + min(pixels2[1])) / 2) #Getting the Center of the Marker
circle_centerY = int((max(pixels2[0]) + min(pixels2[0])) / 2)

image_centerX = int((max(pixels[1]) + min(pixels[1])) / 2) #Getting the Center of the Image
image_centerY = int((max(pixels[0]) + min(pixels[0])) / 2) #The center of the image, by sense, marks the location of the UGV based on the picture

x_distance = ((circle_centerX-image_centerX)**2 + (circle_centerY-image_centerY)**2)**(1/2) #Euclidean Distance between the UGV and the Red Marker based on a 2D space (x-coordinates)

print(len(pixels[0])) 
print(len(pixels2[0]))
#print("Distance between the UGV and the Red Marker: ", x_distance)

#canvas1 = np.zeros((383,512,1), dtype = 'uint8') #For checking purposes only
#canvas2 = np.zeros((383,512,1), dtype = 'uint8') #For checking purposes only

'''for x in range(len(pixels2[1])):
    canvas[pixels2[0][x],pixels2[1][x]] = (255) #[0] ug [1] ang arrangement.'''

#canvas1 = cv2.circle(canvas1,(circle_centerX,circle_centerY), 20, (255,0,0), 2)
#canvas2 = cv2.circle(canvas2,(image_centerX,image_centerY), 20, (255,255,255), 2)


#cv2.imshow("Canvas1",canvas1)
#cv2.imshow("Canvas2",canvas2)
cv2.imshow("Orig Image", img)
cv2.imshow("Detected Red",output_img) 
cv2.waitKey(0)