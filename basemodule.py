# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img1 = cv2.imread('cup1.png',1)

# # Initiate ORB detector
# orb = cv2.ORB_create()
# # find the keypoints with ORB
# kp = orb.detect(img1,None)
# # compute the descriptors with ORB
# kp, des = orb.compute(img1, kp)
# # draw only keypoints location,not size and orientation
# img_points = cv2.drawKeypoints(img1, kp, None,color=(0,255,0), flags=0)
# #img_points1 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img1), plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('book1.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('images.jpeg',cv2.IMREAD_GRAYSCALE) # trainImage 
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
plt.imshow(img3)
plt.show()