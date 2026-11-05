from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np
from time import sleep

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Matice homografie H (převod z pixelů [u, v] na reálné X, Y středu KRUHU v mm)
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

def move_home():
    print("Stěhování do domovské polohy (Home)...")
    move_bot(Home_position)

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

def transform_pixel_to_robot(point_pixel, H_mat):
    """
    Převede pixelové souřadnice středu base na reálné X, Y robota v mm.
    """
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return x_mm, y_mm

def get_gripper_pose_for_circle_target(x_target_m, y_target_m, z_circle_target_m):
    """
    Spočítá matici T_gripper (4x4) pro požadovanou pozici STŘEDU KRUHU [x, y, z] v metrech.
    Využívá známý offset v osách gripperu: +135 mm v X_g, -30 mm v Z_g.
    """
    # Orientace gripperu
    R_gripper = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    # Offset v lokalním souřadném systému gripperu (v metrech)
    p_circle_gripper = np.array([0.135, 0.0, -0.030])
    
    # Přepočet offsetu do souřadného systému robota
    v_robot_offset = R_gripper @ p_circle_gripper  # Vychází [0.135, 0.0, 0.030]
    
    # Pozice příruby robota
    p_gripper = np.array([
        x_target_m - v_robot_offset[0],
        y_target_m - v_robot_offset[1],
        z_circle_target_m - v_robot_offset[2]
    ])
    
    T_gripper = np.eye(4)
    T_gripper[0:3, 0:3] = R_gripper
    T_gripper[0:3, 3] = p_gripper
    return T_gripper


# ==========================================
# HLAVNÍ SEKVENCE POHYBU ROBOTA
# ==========================================

# 1. Přesun do domovské pozice a snímek z kamery
move_home()
sleep(0.5)

img = robot.grab_image()

if img is not None:
    # 2. Nalezení středu base v pixelech
    base_center_px = get_base_center_px(img)
    
    if base_center_px is not None:
        # 3. Přepočet z pixelů na cílovou pozici středu desky (v metrech)
        x_base_mm, y_base_mm = transform_pixel_to_robot(base_center_px, H)
        x_base_m = x_base_mm / 1000.0
        y_base_m = y_base_mm / 1000.0
        
        print(f"Střed base v pixelech: {base_center_px}")
        print(f"Střed base v souřadnicích robota: X = {x_base_m:.4f} m, Y = {y_base_m:.4f} m")
        
        # 4. Pohyb: Střed KRUHU na tyčce přesně nad střed desky do výšky Z = 30 cm (0.30 m)
        z_target_1 = 0.30
        print(f"Pohyb středu KRUHU nad střed base do výšky Z = {z_target_1 * 100:.0f} cm X = {x_base_m}m a Y = {y_base_m}m.")
        
        # Čekání 2 sekundy
        print("Čekání 2 sekundy...")
        sleep(2.0)
        
        T_target_z30 = get_gripper_pose_for_circle_target(x_base_m, y_base_m, z_target_1)
        move_bot(T_target_z30)
        
        # Čekání 2 sekundy
        print("Čekání 2 sekundy...")
        sleep(2.0)
        
        # 5. Pohyb: Střed KRUHU na tyčce klesne přesně nad střed desky do výšky Z = 14 cm (0.14 m)
        z_target_2 = 0.14
        print(f"Klesání středu KRUHU nad střed base do výšky Z = {z_target_2 * 100:.0f} cm...")
        T_target_z14 = get_gripper_pose_for_circle_target(x_base_m, y_base_m, z_target_2)
        move_bot(T_target_z14)

else:
    print("Nepodařilo se získat snímek z kamery.")

# Ukončení připojení
robot.stop_robot()
robot.close_connection()