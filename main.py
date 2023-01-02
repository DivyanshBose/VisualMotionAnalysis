import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture("C:\\VisualMotionAnalysis\\Videos\\Sample3.mp4")
# cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (750, 750))

    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 255, 0), 2, 2))

    h, w, c = img.shape
    opImg = np.zeros([h, w, c])
    opImg.fill(255)
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 255, 0), 2, 2))
    # print(results.pose_landmarks)

    lst = list(mp_pose.POSE_CONNECTIONS)
    lst1 = []
    lst2 = []
    for i in lst:
        lst1.append(i[0])
        lst2.append(i[1])



    cv2.imshow("Pose Estimation", img)

    cv2.imshow("Extracted Pose", opImg)

    k = cv2.waitKey(1)
    if k == 27:
        break
#print(lst1)
#print(lst2)
df = pd.DataFrame(lst1, columns=["F1"])
df = df.assign(F2=lst2)
print(df)
# Release the VideoCapture object.
cap.release()

# Close the windows.
cv2.destroyAllWindows()

