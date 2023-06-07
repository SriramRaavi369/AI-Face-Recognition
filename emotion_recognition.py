import cv2
from  facial_emotion_recognition import EmotionRecognition
er=EmotionRecognition(device='cpu')
cam=cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frame=er.recognise_emotion(frame,return_type="BGR")
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
