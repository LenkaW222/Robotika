import cv2
import numpy as np
from time import sleep, time
from ctu_mitsubishi import Rv6s

# ==========================================
# KONSTANTY A HOMOGRAFIE
# ==========================================
H = np.array([
    [-0.000100, -0.261713, 842.805323],
    [-0.267645, -0.000507, 422.676470],
    [-0.000004,  0.000005,   1.000000]
])

Home_position = np.array([
    [1,  0,  0,  0.4],
    [0, -1,  0, -0.3],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]
])

# Predikční prodleva robota na 3.0 sekundy
ROBOT_DELAY = 3.0

# ==========================================
# POMOCNÉ FUNKCE
# ==========================================
def transform_pixel_to_robot(point_pixel, H_mat):
    """Převede bod [u, v] z pixelů na reálné [X, Y] robota v milimetrech[cite: 4, 5]."""
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])

def move_bot(robot, position):
    """
    Provede inverzní kinematiku. 
    Vrací True při úspěchu, False pokud IK selže.
    """
    if position is None:
        return False
        
    q0 = robot.get_q()
    qs = robot.ik(position)
    
    if qs is None or len(qs) == 0:
        print(f"[CHYBA IK] Pro cílovou pozici X={position[0,3]:.3f}m, Y={position[1,3]:.3f}m, Z={position[2,3]:.3f}m neexistuje platné řešení IK!")
        return False

    qnew = min(qs, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)
    return True

def move_home(robot):
    """Bezpečný návrat robota do domovské pozice[cite: 4, 5]."""
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)
    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift[2, 3] = 0.32
        if not move_bot(robot, T_lift):
            return False
    return move_bot(robot, Home_position)

def get_gripper_pose_for_circle_target(x_target_m, y_target_m, z_target_m):
    """Spočítá matici T_gripper pro nástroj[cite: 4, 5]."""
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
    """Detekuje ArUco značky a vrátí střed base (px) a směrový vektor UP v prostoru robota[cite: 5]."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        return None, None
        
    ids_flat = ids.flatten()
    markers = {marker_id: corners[idx].reshape(4, 2) for idx, marker_id in enumerate(ids_flat)}

    # Střed base v px[cite: 3, 5]
    if 1 in markers and 2 in markers:
        base_center_px = (np.mean(markers[1], axis=0) + np.mean(markers[2], axis=0)) / 2.0
    else:
        target_id = 1 if 1 in markers else (2 if 2 in markers else None)
        if target_id is None:
            return None, None
        pts = markers[target_id]
        c_marker = np.mean(pts, axis=0)
        u_vec = pts[1] - pts[0]
        v_vec = pts[3] - pts[0]
        side_length_px = (np.linalg.norm(u_vec) + np.linalg.norm(v_vec)) / 2.0
        u_hat, v_hat = u_vec / np.linalg.norm(u_vec), v_vec / np.linalg.norm(v_vec)
        offset_factor = 37.5 / 40.0
        
        if target_id == 1:
            offset_vector_px = (u_hat * offset_factor * side_length_px) + (v_hat * offset_factor * side_length_px)
        else:
            offset_vector_px = (-u_hat * offset_factor * side_length_px) + (-v_hat * offset_factor * side_length_px)
        base_center_px = c_marker + offset_vector_px

    # Vektor orientace "UP" desky[cite: 5]
    ref_id = 1 if 1 in markers else (2 if 2 in markers else None)
    pts = markers[ref_id]
    p_bl_mm = transform_pixel_to_robot(pts[3], H)
    p_tl_mm = transform_pixel_to_robot(pts[0], H)
    p_br_mm = transform_pixel_to_robot(pts[2], H)
    p_tr_mm = transform_pixel_to_robot(pts[1], H)

    vec_up = ((p_tl_mm - p_bl_mm) + (p_tr_mm - p_br_mm)) / 2.0
    dir_up_robot = vec_up / -np.linalg.norm(vec_up)

    return base_center_px, dir_up_robot

def fit_circle_least_squares(points):
    """Proloží body v px nejlepší kružnicí[cite: 3]."""
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = -a / 2.0, -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    return np.array([xc, yc]), radius

def calculate_rotation(points, timestamps, center_pt):
    """Spočítá úhlovou rychlost w (rad/s) rotace desky[cite: 3]."""
    pts = np.array(points)
    xc, yc = center_pt
    angles = np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc)
    angles_unwrapped = np.unwrap(angles)
    
    delta_angles = np.diff(angles_unwrapped)
    delta_times = np.diff(timestamps)
    
    mean_w_rad_s = np.mean(delta_angles / delta_times)
    direction = "CW (po směru)" if mean_w_rad_s > 0 else "CCW (proti směru)"
    speed_deg_s = abs(np.degrees(mean_w_rad_s))
    period_sec = (2 * np.pi) / abs(mean_w_rad_s) if mean_w_rad_s != 0 else 0
    
    return mean_w_rad_s, speed_deg_s, period_sec, direction


# ==========================================
# HLAVNÍ SKRIPT
# ==========================================
if __name__ == "__main__":
    robot = Rv6s(debug=False)
    robot.initialize()

    # KROK 1: Home position
    if not move_home(robot):
        print("Chyba při startu do Home position. Ukončuji program.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # KROK 2: Snímání po dobu 20 s[cite: 3]
    base_centers_px = []
    timestamps = []
    base_center_pts_mm = []  # PŘIDÁNO: Souřadnice středu base v mm robota
    start_path_pts_mm = []   # Souřadnice bodu (+7cm) v prostoru robota
    
    total_duration = 20.0
    interval = 1.0
    
    print(f"\n--- [KROK 2] Snímání otáčení stolku po dobu {total_duration}s ---")
    t_start_scan = time()
    last_grab = 0

    while (time() - t_start_scan) < total_duration:
        curr_t = time() - t_start_scan
        if curr_t - last_grab >= interval:
            last_grab = curr_t
            img = robot.grab_image()
            if img is not None:
                c_px, d_up = get_base_info(img)
                if c_px is not None and d_up is not None:
                    base_centers_px.append(c_px)
                    timestamps.append(curr_t)
                    
                    c_mm = transform_pixel_to_robot(c_px, H)
                    base_center_pts_mm.append(c_mm)  # Uložení středu desky
                    
                    start_path_mm = c_mm + d_up * 70.0  # +70 mm[cite: 5]
                    start_path_pts_mm.append(start_path_mm)
                    
                    print(f"[{curr_t:.1f}s] Střed base px: [{c_px[0]:.1f}, {c_px[1]:.1f}]")
        sleep(0.05)

    if len(base_centers_px) < 4:
        print("Chyba: Nedostatek bodů pro analýzu rotace.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # Vyhodnocení rotace středu desky[cite: 3]
    rot_center_px, radius_px = fit_circle_least_squares(base_centers_px)
    w_rad_s, speed_deg_s, period_s, direction = calculate_rotation(base_centers_px, timestamps, rot_center_px)
    rot_center_mm = transform_pixel_to_robot(rot_center_px, H)

    # KROK 3: Počáteční čas t0 a přesun NAD STŘED ROTACE
    print("\n--- [KROK 3] Zapsání času t0 a přesun NAD STŘED ROTACE (Z = 32 cm) ---")
    t0 = time()
    
    rot_center_m = rot_center_mm / 1000.0
    T_rot_center_z32 = get_gripper_pose_for_circle_target(rot_center_m[0], rot_center_m[1], 0.32)
    
    if not move_bot(robot, T_rot_center_z32):
        print("Pohyb nad střed rotace selhal na IK. Zrušen další pohyb.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # PREDIKČNÍ MODELY Z PRVNÍ FÁZE:
    dt_last = (t0 - t_start_scan) - timestamps[-1]

    # Model A: ZAČÁTEK TYČE (+7 cm)
    last_start_path_mm = start_path_pts_mm[-1]
    vec_start_from_rot_mm = last_start_path_mm - rot_center_mm
    radius_start_mm = np.linalg.norm(vec_start_from_rot_mm)
    angle_start_t0 = np.arctan2(vec_start_from_rot_mm[1], vec_start_from_rot_mm[0]) + w_rad_s * dt_last

    def get_predicted_start_path_m(dt_seconds):
        """Spočítá přesnou pozici začátku tyče (+7 cm od středu) v čase t0 + dt_seconds."""
        angle_future = angle_start_t0 + w_rad_s * dt_seconds
        start_future_mm = rot_center_mm + radius_start_mm * np.array([np.cos(angle_future), np.sin(angle_future)])
        return start_future_mm / 1000.0

    # Model B: STŘED BASE (PŘIDÁNO)
    last_base_center_mm = base_center_pts_mm[-1]
    vec_base_from_rot_mm = last_base_center_mm - rot_center_mm
    radius_base_mm = np.linalg.norm(vec_base_from_rot_mm)
    angle_base_t0 = np.arctan2(vec_base_from_rot_mm[1], vec_base_from_rot_mm[0]) + w_rad_s * dt_last

    def get_predicted_base_center_m(dt_seconds):
        """Spočítá přesnou pozici středu desky (base) v čase t0 + dt_seconds."""
        angle_future = angle_base_t0 + w_rad_s * dt_seconds
        base_future_mm = rot_center_mm + radius_base_mm * np.array([np.cos(angle_future), np.sin(angle_future)])
        return base_future_mm / 1000.0

    # KROK 4: Pohyb nad začátek cesty v čase t0 + 10s (s predikcí 3s) na Z = 32 cm
    print("--- [KROK 4] Přesun NAD ZAČÁTEK CESTY v t0 + 10s (Z = 32 cm) ---")
    
    t_target_krok4 = 10.0 + ROBOT_DELAY
    pos_krok4_m = get_predicted_start_path_m(t_target_krok4)
    T_start_z32 = get_gripper_pose_for_circle_target(pos_krok4_m[0], pos_krok4_m[1], 0.32)

    time_to_wait = 10.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    if not move_bot(robot, T_start_z32):
        print("Pohyb v KROKU 4 selhal na IK. Ruším všechny další pohyby!")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # KROK 5: Klesání na Z = 22 cm v čase t0 + 12s (s predikcí 3s)
    print("--- [KROK 5] Klesání na Z = 22 cm v t0 + 12s ---")
    
    t_target_krok5 = 12.0 + ROBOT_DELAY
    pos_krok5_m = get_predicted_start_path_m(t_target_krok5)
    T_start_z22 = get_gripper_pose_for_circle_target(pos_krok5_m[0], pos_krok5_m[1], 0.22)

    time_to_wait = 12.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    if not move_bot(robot, T_start_z22):
        print("Pohyb v KROKU 5 selhal na IK. Ruším všechny další pohyby!")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # KROK 6 (PŘIDÁNO): Pohyb nad predikovaný střed base v čase t0 + 15s na Z = 15 cm
    print("--- [KROK 6] Přesun NAD STŘED BASE v t0 + 15s (Z = 15 cm) ---")
    
    t_target_krok6 = 15.0 + ROBOT_DELAY
    pos_krok6_m = get_predicted_base_center_m(t_target_krok6)
    T_base_z15 = get_gripper_pose_for_circle_target(pos_krok6_m[0], pos_krok6_m[1], 0.15)

    time_to_wait = 15.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)

    if not move_bot(robot, T_base_z15):
        print("Pohyb v KROKU 6 selhal na IK. Ruším všechny další pohyby!")
        robot.stop_robot()
        robot.close_connection()
        exit()

    print("Všechny pohyby byly úspěšně dokončeny.")
    print("Jupí")

    # Ukončení robota
    robot.stop_robot()
    robot.close_connection()
    