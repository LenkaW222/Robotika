import cv2
import numpy as np
from time import sleep, time
from ctu_mitsubishi import Rv6s

# ==========================================
# KONSTANTY A HOMOGRAFIE
# ==========================================
# Homografická matice pro převod z px do mm robota[cite: 4, 5]
H = np.array([
    [-0.000100, -0.261713, 842.805323],
    [-0.267645, -0.000507, 422.676470],
    [-0.000004,  0.000005,   1.000000]
])

# Domovská pozice robota[cite: 4, 5]
Home_position = np.array([
    [1,  0,  0,  0.4],
    [0, -1,  0, -0.3],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]
])

# Odhadovaná prodleva robota mezi odesláním a dokončením pohybu (sekundy)
ROBOT_DELAY = 2.2

# ==========================================
# POMOCNÉ FUNKCE PRO ROBOTA A GEOMETRII
# ==========================================
def transform_pixel_to_robot(point_pixel, H_mat):
    """Převede bod [u, v] z pixelů na reálné [X, Y] robota v milimetrech[cite: 4, 5]."""
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])

def move_bot(robot, position):
    """Provede inverzní kinematiku a posune robota do požadované matice T[cite: 4, 5]."""
    if position is None:
        robot.stop_robot()
        robot.close_connection()
        exit()
    q0 = robot.get_q()
    qs = robot.ik(position)
    qnew = min(qs, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)

def move_home(robot):
    """Bezpečný návrat robota do domovské pozice[cite: 4, 5]."""
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)
    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift[2, 3] = 0.32
        move_bot(robot, T_lift)
    move_bot(robot, Home_position)

def get_gripper_pose_for_circle_target(x_target_m, y_target_m, z_target_m):
    """Spočítá T_gripper pro požadovanou pozici středu kruhu nástroje[cite: 4, 5]."""
    R_gripper = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    # Offset nástroje (tyčky) vůči přírubě v metrech (+135 mm v X_g, +30 mm v Z_g)[cite: 4, 5]
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
    """Detekuje ArUco značky, spočítá střed base v px a směrový vektor UP v prostoru robota[cite: 5]."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        return None, None
        
    ids_flat = ids.flatten()
    markers = {marker_id: corners[idx].reshape(4, 2) for idx, marker_id in enumerate(ids_flat)}

    # Střed base[cite: 3, 5]
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
    """Proloží body v px nejlepší kružnicí pro nalezení středu otáčení stolku[cite: 3]."""
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = -a / 2.0, -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    return np.array([xc, yc]), radius

def calculate_rotation(points, timestamps, center_pt):
    """Vypočítá úhlovou rychlost stolku w (rad/s), rychlost v deg/s, periodu a směr[cite: 3]."""
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

    # KROK 1: Přesun do domovské polohy
    move_home(robot)

    # KROK 2: Snímání po dobu 20 s a výpis údajů o rotaci[cite: 3]
    base_centers_px = []
    timestamps = []
    dir_up_vectors = []
    
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
                if c_px is not None:
                    base_centers_px.append(c_px)
                    timestamps.append(curr_t)
                    dir_up_vectors.append(d_up)
                    print(f"[{curr_t:.1f}s] Střed base (px): [{c_px[0]:.1f}, {c_px[1]:.1f}]")
        sleep(0.05)

    if len(base_centers_px) < 4:
        print("Chyba: Nedostatek bodů pro analýzu rotace.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # Vyhodnocení parametrů rotace stolku[cite: 3]
    rot_center_px, radius_px = fit_circle_least_squares(base_centers_px)
    w_rad_s, speed_deg_s, period_s, direction = calculate_rotation(base_centers_px, timestamps, rot_center_px)
    
    # Přepočet do souřadného systému robota
    rot_center_mm = transform_pixel_to_robot(rot_center_px, H)
    rot_center_cm = rot_center_mm / 10.0
    
    # Odhad poloměru rotace středu desky v cm
    p0_mm = transform_pixel_to_robot(rot_center_px, H)
    p1_mm = transform_pixel_to_robot(rot_center_px + np.array([radius_px, 0]), H)
    radius_cm = np.linalg.norm(p1_mm - p0_mm) / 10.0

    print("\n" + "="*50)
    print(" VÝSLEDKY DETEKCE A ANALÝZY ROTACE")
    print("="*50)
    print(f" Čas jedné rotace (perioda): {period_s:.2f} s")
    print(f" Rychlost rotace:             {speed_deg_s:.2f} °/s")
    print(f" Směr rotace:                {direction}")
    print(f" Vzdálenost středu base od středu otáčení: {radius_cm:.2f} cm")
    print(f" Pozice středu otáčení (px): [{rot_center_px[0]:.1f}, {rot_center_px[1]:.1f}]")
    print(f" Pozice středu otáčení (robot): X = {rot_center_cm[0]:.2f} cm, Y = {rot_center_cm[1]:.2f} cm")
    print("="*50 + "\n")

    # KROK 3: Zapsání počátečního času t0 a přesun NAD STŘED ROTACE v Z = 32 cm
    print("--- [KROK 3] Zapsání času t0 a přesun NAD STŘED ROTACE (Z = 32 cm) ---")
    t0 = time()
    
    rot_center_m = rot_center_mm / 1000.0
    T_rot_center_z32 = get_gripper_pose_for_circle_target(rot_center_m[0], rot_center_m[1], 0.32)
    move_bot(robot, T_rot_center_z32)

    # Inicializace úhlu desky pro budoucí predikci
    last_base_px = base_centers_px[-1]
    angle_t_last = np.arctan2(last_base_px[1] - rot_center_px[1], last_base_px[0] - rot_center_px[0])
    dt_last = (t0 - t_start_scan) - timestamps[-1]
    angle_t0 = angle_t_last + w_rad_s * dt_last
    mean_dir_up = np.mean(dir_up_vectors, axis=0)

    def get_predicted_base_center_mm(dt_seconds):
        """Vypočítá pozici středu desky (v mm robota) v čase t0 + dt_seconds s predikcí rotace."""
        angle_future = angle_t0 + w_rad_s * dt_seconds
        base_future_px = rot_center_px + radius_px * np.array([np.cos(angle_future), np.sin(angle_future)])
        return transform_pixel_to_robot(base_future_px, H)

    # KROK 4: Pohyb nad začátek cesty (+7 cm od středu desky) v t = 10s na Z = 32 cm
    print("--- [KROK 4] Přesun NAD ZAČÁTEK CESTY (t0 + 10s + DELAY) na Z = 32 cm ---")
    
    # Cílový čas dojezdu je t0 + 10s, predikujeme pozici s offsetem prodlevy robota
    t_target_krok4 = 10.0 + ROBOT_DELAY
    base_center_krok4_mm = get_predicted_base_center_mm(t_target_krok4)
    start_path_krok4_m = (base_center_krok4_mm + mean_dir_up * 70.0) / 1000.0  # +7 cm[cite: 5]

    T_start_z32 = get_gripper_pose_for_circle_target(start_path_krok4_m[0], start_path_krok4_m[1], 0.32)

    # Čekání na správný okamžik pro odeslání příkazu
    time_to_wait = 10.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    move_bot(robot, T_start_z32)

    # KROK 5: Klesání na Z = 22 cm nad začátkem cesty v t = 12s a provedení KRUHU tyčkou
    print("--- [KROK 5] Klesání na Z = 22 cm (t0 + 12s + DELAY) a kroužení tyčkou ---")
    
    t_target_krok5 = 12.0 + ROBOT_DELAY
    base_center_krok5_mm = get_predicted_base_center_mm(t_target_krok5)
    start_path_krok5_m = (base_center_krok5_mm + mean_dir_up * 70.0) / 1000.0

    T_start_z22 = get_gripper_pose_for_circle_target(start_path_krok5_m[0], start_path_krok5_m[1], 0.22)

    time_to_wait = 12.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    move_bot(robot, T_start_z22)

    # --- PROVEDENÍ KRUHU ZA HNOU TYČKOU ---
    # Generování kružnice o poloměru R_hoop = 30 mm nad rotující deskou
    print("Provádím kruhový pohyb tyčkou po trajektorii...")
    r_hoop_m = 0.03  # 3 cm poloměr kroužení tyčkou
    num_points = 12
    hoop_duration = 6.0  # Doba trvání opisu kruhu v sekundách
    
    t_circle_start = time() - t0

    for i in range(num_points):
        step_progress = i / float(num_points)
        phi = 2 * np.pi * step_progress
        
        # Čas dojezdu daného bodu kružnice
        current_step_time = t_circle_start + step_progress * hoop_duration + ROBOT_DELAY
        
        # Aktualizovaný střed desky v čase průjezdu bodu
        base_current_mm = get_predicted_base_center_mm(current_step_time)
        center_path_m = (base_current_mm + mean_dir_up * 70.0) / 1000.0
        
        # Bod na kružnici vztažený k rotujícímu středu cesty
        x_circle = center_path_m[0] + r_hoop_m * np.cos(phi)
        y_circle = center_path_m[1] + r_hoop_m * np.sin(phi)
        z_circle = 0.22
        
        T_hoop_step = get_gripper_pose_for_circle_target(x_circle, y_circle, z_circle)
        move_bot(robot, T_hoop_step)

    print("\nKruhový pohyb úspěšně dokončen.")

    # Ukončení robota
    robot.stop_robot()
    robot.close_connection()