from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np
import time 

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

Home_position = np.array(
    [[1,  0,  0, 0.4],
     [0, -1,  0, -0.3],
     [0,  0, -1, 0.4],
     [0,  0,  0, 1.0]]
)


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
        time.sleep(0.05)
        print("steps(" + str(i) + "/" + str(steps)+"), ", end = "")
        print(qmove)
    print("moved", end = "\n\n")

move_bot(Home_position)

start_time= time.time()
for i in range (30):

    img = robot.grab_image()
    if img is not None:
        os.makedirs('photos_rotating', exist_ok=True)

        success = cv2.imwrite('photos_rotating/img'+str(i)+'.png', img)
        print(str(i) + ": " +str(success))

        cas = time.time()-start_time

        with open('photos_rotating/positions'+str(i) + '.txt', 'w') as f:
            f.write(str(cas))

    time.sleep(0.4521354)

    position = None


robot.stop_robot()
robot.close_connection()