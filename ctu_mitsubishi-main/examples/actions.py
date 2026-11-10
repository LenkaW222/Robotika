from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np


# Pokus o vytvoreni erarnich funkci-> zatim nefunguje, protoze se pouziva classa robota, ktera zde neni definovana

robot = Rv6s(debug=False)

def move_home():
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)

    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift [2, 3] = 0.32

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

    steps = 2

    for i in range(steps):
        qmove = q0 - (q0-qnew)*(i/(steps-1))
        robot.move_to_q(qmove)
        robot.wait_for_motion_stop()
        sleep(0.05)
        print("steps(" + str(i) + "/" + str(steps)+"), ", end = "")
        print(qmove)
    print("moved", end = "\n\n")


def transform_pixel_to_robot(point_pixel, H_mat):
    """
    Převede bod [u, v] z pixelů na reálné [X, Y] robota v milimetrech.
    """
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])




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
    p_circle_gripper = np.array([0.135, 0.0, 0.030])
    
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



def get_base_center_robust(img):
    """
    Detekuje ArUco značky v obraze:
    1. Pokud vidí obě (ID1 i ID2), spočítá střed jako jejich geometrický průměr.
    2. Pokud vidí pouze jednu (ID1 nebo ID2), dopočítá střed base z orientace rohu 
       a offsetu 37.5 mm z výkresu.
    
    Vrací: (base_center_px, detected_markers_dict)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        print("Chyba: Nebyla detekována žádná ArUco značka.")
        return None, {}
        
    ids_flat = ids.flatten()
    markers = {}
    
    for idx, marker_id in enumerate(ids_flat):
        pts = corners[idx].reshape(4, 2)
        markers[marker_id] = pts

    base_center_px = None

    # PŘÍPAD A: Detekovány obě značky (ID1 i ID2)
    if 1 in markers and 2 in markers:
        c1 = np.mean(markers[1], axis=0)
        c2 = np.mean(markers[2], axis=0)
        base_center_px = (c1 + c2) / 2.0
        print("Střed base vypočítán z obou ArUco značek (ID1 i ID2).")
        
    # PŘÍPAD B: Detekována pouze jedna značka (ID1 nebo ID2)
    else:
        target_id = 1 if 1 in markers else (2 if 2 in markers else None)
        
        if target_id is not None:
            pts = markers[target_id]
            c_marker = np.mean(pts, axis=0)
            
            # Vektory orientace stran značky
            u_vec = pts[1] - pts[0]  # Osa X značky
            v_vec = pts[3] - pts[0]  # Osa Y značky
            
            side_length_px = (np.linalg.norm(u_vec) + np.linalg.norm(v_vec)) / 2.0
            u_hat = u_vec / np.linalg.norm(u_vec)
            v_hat = v_vec / np.linalg.norm(v_vec)
            
            # Offset středu base od středu značky: (37.5 mm / 40.0 mm) * délka strany v px
            offset_factor = 37.5 / 40.0
            
            if target_id == 1:
                # ID1 (vlevo nahoře): střed base leží vpravo (+X) a dolů (+Y)
                offset_vector_px = (u_hat * offset_factor * side_length_px) + (v_hat * offset_factor * side_length_px)
            else:
                # ID2 (vpravo dole): střed base leží vlevo (-X) a nahoru (-Y)
                offset_vector_px = (-u_hat * offset_factor * side_length_px) + (-v_hat * offset_factor * side_length_px)
                
            base_center_px = c_marker + offset_vector_px
            print(f"Střed base dopočten z jediné ArUco značky ID{target_id}.")

    return base_center_px, markers