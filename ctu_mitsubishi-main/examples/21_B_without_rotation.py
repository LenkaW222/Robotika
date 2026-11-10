from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np
import time
import actions

# Matice homografie H (převod z pixelů [u, v] na reálné X, Y středu v mm)
H = np.array([
    [-0.000100, -0.261713, 842.805323],
    [-0.267645, -0.000507, 422.676470],
    [-0.000004, 0.000005, 1.000000]])

Home_position = np.array(
    [[1,  0,  0, 0.4],
     [0, -1,  0, -0.3],
     [0,  0, -1, 0.4],
     [0,  0,  0, 1.0]]
)


def move_home():
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)

    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift[2, 3] = 0.32
        move_bot(T_lift)
    move_bot(Home_position)


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

    qmove = qnew
    robot.move_to_q(qmove)
    robot.wait_for_motion_stop(poll_interval=0.01)
    #sleep(0.05)
    #print("steps(" + str(i) + "/" + str(steps)+"), ", end="")
    print(qmove)

def plan_motion(T, q0):
    Qplan = []
    qold = q0
    for i in range(len(T)):
        qs = robot.ik(T[i])
        qnew = min(qs, key=lambda q: np.linalg.norm(q-qold))

        qold = qnew
        Qplan.append(qold)

    return Qplan




def transform_pixel_to_robot(point_pixel, H_mat):
    """
    Převede bod [u, v] z pixelů na reálné [X, Y] robota v milimetrech.
    """
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])


def get_gripper_pose_for_circle_target(x_target_m, y_target_m, z_target_m):
    """
    Spočítá matici T_gripper (4x4) pro požadovanou pozici nástroje [x, y, z] v metrech.
    """
    R_gripper = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    p_circle_gripper = np.array([0.135, 0.0, 0.030])
    v_robot_offset = R_gripper @ p_circle_gripper
    
    p_gripper = np.array([
        x_target_m - v_robot_offset[0],
        y_target_m - v_robot_offset[1],
        z_target_m - v_robot_offset[2]
    ])
    
    T_gripper = np.eye(4)
    T_gripper[0:3, 0:3] = R_gripper
    T_gripper[0:3, 3] = p_gripper
    return T_gripper


def get_base_info(img):
    """
    Detekuje ArUco značky a spočítá:
    1. Střed base v pixelech.
    2. Jednotkový směrový vektor "nahoru v base" v souřadnicích robota.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        print("Chyba: Nebyla detekována žádná ArUco značka.")
        return None, None
        
    ids_flat = ids.flatten()
    markers = {}
    
    for idx, marker_id in enumerate(ids_flat):
        pts = corners[idx].reshape(4, 2)
        markers[marker_id] = pts

    base_center_px = None
    dir_up_robot = None

    # --- 1. Výpočet středu desky ---
    if 1 in markers and 2 in markers:
        c1 = np.mean(markers[1], axis=0)  # ID1 (vlevo nahoře dle výkresu)
        c2 = np.mean(markers[2], axis=0)  # ID2 (vpravo dole dle výkresu)
        base_center_px = (c1 + c2) / 2.0
        print("Střed base vypočítán z obou ArUco značek (ID1 i ID2).")
    else:
        target_id = 1 if 1 in markers else (2 if 2 in markers else None)
        if target_id is not None:
            pts = markers[target_id]
            c_marker = np.mean(pts, axis=0)
            
            u_vec = pts[1] - pts[0]
            v_vec = pts[3] - pts[0]
            side_length_px = (np.linalg.norm(u_vec) + np.linalg.norm(v_vec)) / 2.0
            u_hat = u_vec / np.linalg.norm(u_vec)
            v_hat = v_vec / np.linalg.norm(v_vec)
            
            offset_factor = 37.5 / 40.0
            if target_id == 1:
                # ID1 (vlevo nahoře) -> střed je vpravo (+u) a dolů (+v v pixelech)
                offset_vector_px = (u_hat * offset_factor * side_length_px) + (v_hat * offset_factor * side_length_px)
            else:
                # ID2 (vpravo dole) -> střed je vlevo (-u) a nahoru (-v v pixelech)
                offset_vector_px = (-u_hat * offset_factor * side_length_px) + (-v_hat * offset_factor * side_length_px)
                
            base_center_px = c_marker + offset_vector_px
            print(f"Střed base dopočten z jediné ArUco značky ID{target_id}.")

    # --- 2. Výpočet směrového vektoru "NAHORU v base" ---
    # Na výkresu směřuje osa desky od spodní hrany k horní.
    # Vektor z dolního rohu značky (BL/BR) do horního rohu (TL/TR):
    ref_id = 1 if 1 in markers else (2 if 2 in markers else None)
    if ref_id is not None:
        pts = markers[ref_id]
        
        p_bl_mm = transform_pixel_to_robot(pts[3], H)  # Bottom-Left
        p_tl_mm = transform_pixel_to_robot(pts[0], H)  # Top-Left
        p_br_mm = transform_pixel_to_robot(pts[2], H)  # Bottom-Right
        p_tr_mm = transform_pixel_to_robot(pts[1], H)  # Top-Right

        # Vektor směřující od dolní hrany k horní hraně značky v reálném prostoru
        vec_up_left = p_tl_mm - p_bl_mm
        vec_up_right = p_tr_mm - p_br_mm
        vec_up = (vec_up_left + vec_up_right) / 2.0
        
        dir_up_robot = vec_up / -np.linalg.norm(vec_up)


    return base_center_px, dir_up_robot

robot = Rv6s(debug=False)
robot.initialize()

move_home()
time.sleep(0.5)

img = robot.grab_image()

if img is not None:
    base_center_px, dir_up_robot = get_base_info(img)
    
    if base_center_px is not None and dir_up_robot is not None:
        x_base_mm, y_base_mm = transform_pixel_to_robot(base_center_px, H)
        x_base_m = x_base_mm / 1000.0
        y_base_m = y_base_mm / 1000.0
        
        # Souřadnice bodu posunutého o 7 cm (0.07 m) směrem nahoru v base
        shift_m = 0.07
        x_shift_m = x_base_m + dir_up_robot[0] * shift_m
        y_shift_m = y_base_m + dir_up_robot[1] * shift_m
        
        print("\n--- DETEKOVANÉ SOUŘADNICE ---")
        print(f"Střed base v prostoru robota:         X = {x_base_m:.4f} m, Y = {y_base_m:.4f} m")
        print(f"Začátek nového obrazce (+7 cm nahoru): X = {x_shift_m:.4f} m, Y = {y_shift_m:.4f} m\n")

        # --- KROK 1: Přesun nad střed base v z = 32 cm ---
        z1 = 0.32
        print(f"[PLÁN 1/5] Přesun NAD STŘED BASE | Cílová pozice: X = {x_base_m * 100:.2f} cm, Y = {y_base_m * 100:.2f} cm, Z = {z1 * 100:.2f} cm")
        T1 = get_gripper_pose_for_circle_target(x_base_m, y_base_m, z1)
        #move_bot(T1)
        #sleep(2.0)

        # --- KROK 2: Přesun nad začátek nové tyče v z = 32 cm ---
        print(f"[PLÁN 2/5] Přesun NAD ZAČÁTEK OBRAZCE (+7 cm) | Cílová pozice: X = {x_shift_m * 100:.2f} cm, Y = {y_shift_m * 100:.2f} cm, Z = {z1 * 100:.2f} cm")
        T2 = get_gripper_pose_for_circle_target(x_shift_m, y_shift_m, z1)
        #move_bot(T2)
        #sleep(2.0)

        # --- KROK 3: Klesání nad začátkem nové tyče na z = 25 cm ---
        z3 = 0.22
        print(f"[PLÁN 3/5] Klesání NAD ZAČÁTKEM OBRAZCE | Cílová pozice: X = {x_shift_m * 100:.2f} cm, Y = {y_shift_m * 100:.2f} cm, Z = {z3 * 100:.2f} cm")
        T3 = get_gripper_pose_for_circle_target(x_shift_m, y_shift_m, z3)
        #move_bot(T3)
        #sleep(2.0)

        # --- KROK 4: Přesun nad střed base a z = 18 cm ---
        z4 = 0.15
        print(f"[PLÁN 4/5] Přesun NAD STŘED BASE | Cílová pozice: X = {x_base_m * 100:.2f} cm, Y = {y_base_m * 100:.2f} cm, Z = {z4 * 100:.2f} cm")
        T4 = get_gripper_pose_for_circle_target(x_base_m, y_base_m, z4)
        #move_bot(T4)
        #sleep(2.0)

        # --- KROK 5: Klesání nad středem base na z = 10 cm ---
        z5 = 0.1
        print(f"[PLÁN 5/5] Klesání NAD STŘEDEM BASE | Cílová pozice: X = {x_base_m * 100:.2f} cm, Y = {y_base_m * 100:.2f} cm, Z = {z5 * 100:.2f} cm")
        T5 = get_gripper_pose_for_circle_target(x_base_m, y_base_m, z5)
        #move_bot(T5)
        #sleep(2.0)
        T = [T1, T2, T3, T4, T5]


        Qplan = plan_motion(T, robot.get_q())

        for i in range(len(Qplan)):
            pred = time.time()
            robot.move_to_q(Qplan[i])
            robot.wait_for_motion_stop(poll_interval=0.01)
            po = time.time()
            print(po - pred)


else:
    print("Nepodařilo se získat snímek z kamery.")

# Ukončení připojení
robot.stop_robot()
robot.close_connection()