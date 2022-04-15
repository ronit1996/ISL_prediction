import cv2
import os

# This file is for generating test images from videos, give path to vide and it will #
# save the test images in the appropriate folder (you have to mention the digit it's saving #

folder_name = 8
video_name = "fab_8"
vid = cv2.VideoCapture("./video/{}.mp4".format(video_name))

count = 1
success = True
while success:
    success, image = vid.read()
    os.makedirs("./myData/{}".format(folder_name), exist_ok=True)
    cv2.imwrite("./myData/{}/fab{}.jpg".format(folder_name, count), image)
    print("saved frame {}".format(count))
    count += 1
    if count > 1000:
        break
