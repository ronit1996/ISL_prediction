import joblib
import os
import cv2
import numpy as np

# local import #
from definitions import image_test

# This file is just for testing single images or a folder of images not for videos #

model = joblib.load("trained_model")
single_test = "yes"
categories = os.listdir("./Indian")

if single_test == "no":

    # The initial for loop in the if statement takes all the images in a folder, predicts them #
    # prints the predictions in the images along with the actual number and then makes a collage #

    for f in os.listdir("internetTest"):
        col = []
        for img in os.listdir("./internetTest/{}".format(f)):
            digit = img.split(".")[0]
            image = cv2.imread("internetTest/{}/{}".format(f, img))
            pred = image_test(image, model)
            result = pred[0]
            res = cv2.resize(image, (200, 200))
            text1 = "Actual number: {}".format(digit)
            cv2.putText(res, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            text2 = "Predicted Number: {}".format(result)
            cv2.putText(res, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            col.append(res)
            collage = np.hstack(col)
            # cv2.imshow("image", res)
        # cv2.imwrite("./internetTest/{}/pred_collage.jpg".format(f), collage)
        cv2.imshow("stack", collage)
        cv2.waitKey(0)
else:

    # The else part is mostly for single files, if you want to test a single image the else part does that #

    image = cv2.imread("./internetTest/fabi_hand/alt_test_1.jpg")
    pred = image_test(image, model)
    print(pred)