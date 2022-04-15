import cv2
from definitions import image_test
import joblib

# This file is for testing videos, the code below can predict signs in a video #

model = joblib.load("trained_model")
cap = cv2.VideoCapture("./internetTest/hand_pred_4.mp4")
while True:
    ret, frame = cap.read()
    pred = image_test(frame, model)
    text = "Predicted Number: {}".format(pred[0])
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    frame_scale = cv2.resize(frame, (540, 960))
    cv2.imshow("frame", frame_scale)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
