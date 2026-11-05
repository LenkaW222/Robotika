from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np
from time import sleep

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Matice homografie H (doplňte vaši vypočítanou matici 3x3)
H = np.array([
    [ 6.47099257e-02, -7.23086873e-01,  8.54855848e+02],
    [-6.74580805e-01,  1.21563085e-02,  4.40229368e+02],
    [ 1.08987336e-04, -1.16147801e-04,  1.00000000e+00]
])

Home_position = np.array(
    [[1,  0,  0, 0.4],
     [0, -1,  0, 0.0],
     [0,  0, -1, 0.4],
     [0,  0,  0, 1.0]]
)

def move_bot(position):
    if position is None:
        robot.stop_robot()
        robot.close_connection()
        exit()

    q0 = robot.get_q()
    print("q0:", q0)
    qs = robot.ik(position)

    qnew = min(qs, key=lambda q: np.linalg.norm(q-q0))

    print("q new: ", end=" ")
    print(qnew)

    steps = 5

    for i in range(steps):
        qmove = q0 - (q0-qnew)*(i/(steps-1))
        robot.move_to_q(qmove)
        robot.wait_for_motion_stop()
        sleep(0.05)
        print("steps(" + str(i) + "/" + str(steps)+"), ", end = "")
        print(qmove)
    print("moved", end = "\n\n")

# Funkce pro návrat do domovské polohy
def move_home():
    print("Stěhování do domovské polohy (Home)...")
    move_bot(Home_position)

# Detekce středu base v pixelech z ArUco značek ID1 a ID2
def get_base_center_px(img):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    centers = {}
    if ids is not None:
        for idx, marker_id in enumerate(ids.flatten()):
            c = corners[idx].reshape(4, 2)
            centers[marker_id] = np.mean(c, axis=0)
            
    if 1 in centers and 2 in centers:
        base_center_px = (centers[1] + centers[2]) / 2.0
        return base_center_px
    else:
        print("Chyba: Nebyly detekovány obě ArUco značky ID1 a ID2!")
        return None

# Převod z obrazových pixelů [u, v] na X, Y robota v milimetrech
def transform_pixel_to_robot(point_pixel, H_mat):
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return x_mm, y_mm


# ==========================================
# HLAVNÍ SEKVENCE POHYBU ROBOTA
# ==========================================

# 1. Přesun do domovské pozice a vyfocení snímku
move_home()
sleep(0.5)

img = robot.grab_image()

if img is not None:
    # 2. Nalezení středu base v obrazu
    base_center_px = get_base_center_px(img)
    
    if base_center_px is not None:
        # 3. Přepočet pixelů na souřadnice robota (X_base, Y_base v metrech)
        x_base_mm, y_base_mm = transform_pixel_to_robot(base_center_px, H)
        x_base_m = x_base_mm / 1000.0
        y_base_m = y_base_mm / 1000.0
        
        print(f"Střed base v pixelech: {base_center_px}")
        print(f"Střed base v souřadnicích robota: X = {x_base_m:.4f} m, Y = {y_base_m:.4f} m")
        
        # Čekání 2 sekundy
        print("Čekání 2 sekundy...")
        sleep(2.0)
        
        # 4. Výpočet pozice příruby gripperu (kruh je posunut o +135 mm v osě X gripperu)
        x_gripper = x_base_m - 0.135
        y_gripper = y_base_m
        
        # 5. Pohyb nad střed base do výšky z = 30 cm (0.30 m)
        print("Pohyb nad střed base do výšky Z = 30 cm...")
        target_z30 = np.array([
            [1,  0,  0, x_gripper],
            [0, -1,  0, y_gripper],
            [0,  0, -1, 0.30],
            [0,  0,  0, 1.0]
        ])
        move_bot(target_z30)
        
        # Čekání 2 sekundy
        print("Čekání 2 sekundy...")
        sleep(2.0)
        
        # 6. Klesání na výšku z = 14 cm (0.14 m)
        print("Klesání nad střed base do výšky Z = 14 cm...")
        target_z14 = np.array([
            [1,  0,  0, x_gripper],
            [0, -1,  0, y_gripper],
            [0,  0, -1, 0.14],
            [0,  0,  0, 1.0]
        ])
        move_bot(target_z14)

else:
    print("Nepodařilo se získat snímek z kamery.")

# Ukončení připojení
robot.stop_robot()
robot.close_connection()
robot.close_connection()