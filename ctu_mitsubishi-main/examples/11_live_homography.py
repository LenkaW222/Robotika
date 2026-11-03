import numpy as np
from ctu_mitsubishi import Rv6s
from time import sleep
import cv2

robot = Rv6s(debug=False)
robot.initialize()

def move_bot(position):
    if position is None:
        robot.stop_robot()
        robot.close_connection()
        exit()

    q0 = robot.get_q()
    print(q0)
    qs = robot.ik(position)

    qnew = min(qs, key=lambda q: np.linalg.norm(q-q0))


    print("q new: ", end=" ")
    print(qnew)

    steps = 2

    for i in range(steps):
        qmove = q0 - (q0-qnew)*(i/(steps-1))
        robot.move_to_q(qmove)
        robot.wait_for_motion_stop()
        sleep(0.1)
        print("steps(" + str(i) + "/" + str(steps)+"), ", end = "")
        print(qmove)
    print("moved", end = "\n\n")

def image_save():
    img = robot.grab_image()
    img = cv2.medianBlur(img, 3)

    circles = cv2.HoughCircles(
        img, 
        cv2.HOUGH_GRADIENT,
        dp = 1,
        minDist=100,
        param1=100,
        param2=40.
        minRadius=100
        maxRadius=100
    )

    if circles is not None:
        circles - np.uint16(np.around(circles))
        x,y,r = circles[0][0]


img_points = np.array()

for i in range (5):
    for j in range(2):
        position = np.array(
        [[1, 0, 0, 0.45 + i*0.04],
        [0, -1, 0, 0+j*0.1],
        [0, 0, -1, 0.1],
        [0, 0, 0, 1]])

        move_bot(position)
        image_save()
        position = None

robot.stop_robot()
robot.close_connection()