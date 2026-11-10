import cv2
import numpy as np
import os
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

# Predikční prodleva robota nastavená přesně na 2.2 sekundy
ROBOT_DELAY = 2.5

# ==========================================
# POMOCNÉ FUNKCE
# ==========================================
def transform_pixel_to_robot(point_pixel, H_mat):
    """Převede bod [u, v] z pixelů na reálné [X, Y] robota v mm."""
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])

def move_bot(robot, position):
    """
    Provede inverzní kinematiku.
    Vrací True při úspěchu, False při selhání IK (bez pádu programu).
    """
    if position is None:
        return False
        
    q0 = robot.get_q()
    qs = robot.ik(position)
    
    if qs is None or len(qs) == 0:
        print(f"[CHYBA IK] Pro pozici X={position[0,3]:.3f}m, Y={position[1,3]:.3f}m, Z={position[2,3]:.3f}m neexistuje řešení. Přeskakuji...")
        return False

    qnew = min(qs, key=lambda q: np.linalg.norm(q - q0))
    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)
    return True

def move_home(robot):
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
    """Spočítá matici T_gripper pro střed kruhu na tyčce."""
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

def get_base_info_px(img):
    """Vrátí střed base v px a směrový vektor 'UP' desky v PIXELECH."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None:
        return None, None
        
    ids_flat = ids.flatten()
    markers = {marker_id: corners[idx].reshape(4, 2) for idx, marker_id in enumerate(ids_flat)}

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

    ref_id = 1 if 1 in markers else (2 if 2 in markers else None)
    pts = markers[ref_id]
    vec_up_px = ((pts[0] - pts[3]) + (pts[1] - pts[2])) / 2.0
    dir_up_px = vec_up_px / np.linalg.norm(vec_up_px)

    return base_center_px, dir_up_px

def fit_circle_least_squares(points):
    """Proloží body v px nejlepší kružnicí."""
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = -a / 2.0, -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    return np.array([xc, yc]), radius

def calculate_rotation(points, timestamps, center_pt):
    """Spočítá úhlovou rychlost w (rad/s) rotace desky."""
    pts = np.array(points)
    xc, yc = center_pt
    angles = np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc)
    angles_unwrapped = np.unwrap(angles)
    
    delta_angles = np.diff(angles_unwrapped)
    delta_times = np.diff(timestamps)
    
    mean_w_rad_s = np.mean(delta_angles / delta_times)
    speed_deg_s = abs(np.degrees(mean_w_rad_s))
    period_sec = (2 * np.pi) / abs(mean_w_rad_s) if mean_w_rad_s != 0 else 0
    
    return mean_w_rad_s, speed_deg_s, period_sec

# ==========================================
# HLAVNÍ SKRIPT
# ==========================================
if __name__ == "__main__":
    robot = Rv6s(debug=False)
    robot.initialize()

    # KROK 1: HOME POSITION
    if not move_home(robot):
        print("Chyba při startu do Home position. Ukončuji program.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    output_dir = "annotated_frames"
    os.makedirs(output_dir, exist_ok=True)

    # KROK 2: SNÍMÁNÍ 20 SEKUND A VYKRESLENÍ ÚDAJŮ
    base_centers_px = []
    timestamps = []
    dir_up_px_list = []
    
    total_duration = 20.0
    interval = 1.0
    
    print(f"\n--- [KROK 2] Snímání otáčení stolku po dobu {total_duration}s ---")
    t_start_scan = time()
    last_grab = 0
    frame_counter = 0

    while (time() - t_start_scan) < total_duration:
        curr_t = time() - t_start_scan
        if curr_t - last_grab >= interval:
            last_grab = curr_t
            img = robot.grab_image()
            if img is not None:
                c_px, d_up_px = get_base_info_px(img)
                if c_px is not None and d_up_px is not None:
                    d_up_px *= -1  # Otočení směru k tyči
                    base_centers_px.append(c_px)
                    timestamps.append(curr_t)
                    dir_up_px_list.append(d_up_px)
                    frame_counter += 1
                    
                    # Zakreslení vektoru do snímku
                    annotated_img = img.copy()
                    start_pt = (int(c_px[0]), int(c_px[1]))
                    end_pt = (
                        int(c_px[0] + d_up_px[0] * 60.0),
                        int(c_px[1] + d_up_px[1] * 60.0)
                    )
                    cv2.circle(annotated_img, start_pt, 5, (255, 0, 0), -1)
                    cv2.arrowedLine(annotated_img, start_pt, end_pt, (0, 0, 255), 3, tipLength=0.3)
                    cv2.putText(annotated_img, f"Smer K TYCI (t={curr_t:.1f}s)", (start_pt[0] + 10, start_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    img_filename = os.path.join(output_dir, f"frame_{frame_counter:02d}_{curr_t:.1f}s.jpg")
                    cv2.imwrite(img_filename, annotated_img)
                    print(f"[{curr_t:.1f}s] Střed base px: [{c_px[0]:.1f}, {c_px[1]:.1f}]")
        sleep(0.05)

    if len(base_centers_px) < 4:
        print("Chyba: Nedostatek bodů pro analýzu rotace.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # Vyhodnocení rotace stolku a přepočty na cm
    rot_center_px, radius_px = fit_circle_least_squares(base_centers_px)
    w_rad_s, speed_deg_s, period_s = calculate_rotation(base_centers_px, timestamps, rot_center_px)
    rot_center_mm = transform_pixel_to_robot(rot_center_px, H)
    rot_center_cm = rot_center_mm / 10.0

    p0_mm = transform_pixel_to_robot(rot_center_px, H)
    p1_mm = transform_pixel_to_robot(rot_center_px + np.array([radius_px, 0]), H)
    radius_cm = np.linalg.norm(p1_mm - p0_mm) / 10.0

    print("\n" + "="*50)
    print(" VÝSLEDKY DETEKCE A ANALÝZY ROTACE")
    print("="*50)
    print(f" Čas jedné rotace (perioda): {period_s:.2f} s")
    print(f" Rychlost rotace:             {speed_deg_s:.2f} °/s")
    print(f" Vzdálenost středu base od středu otáčení: {radius_cm:.2f} cm")
    print(f" Pozice středu otáčení (px): [{rot_center_px[0]:.1f}, {rot_center_px[1]:.1f}]")
    print(f" Pozice středu otáčení (robot): X = {rot_center_cm[0]:.2f} cm, Y = {rot_center_cm[1]:.2f} cm")
    print("="*50 + "\n")

    # KROK 3: ZAPSÁNÍ POČÁTEČNÍHO ČASU t0 A POHYB NAD STŘED ROTACE
    print("--- [KROK 3] Zapsání času t0 a přesun NAD STŘED ROTACE (Z = 32 cm) ---")
    t0 = time()
    
    rot_center_m = rot_center_mm / 1000.0
    T_rot_center_z32 = get_gripper_pose_for_circle_target(rot_center_m[0], rot_center_m[1], 0.32)
    
    if not move_bot(robot, T_rot_center_z32):
        print("Pohyb NAD STŘED ROTACE selhal na IK. Ruším další pohyby!")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # PREDIKČNÍ MODELY
    dt_last = (t0 - t_start_scan) - timestamps[-1]

    last_c_px = base_centers_px[-1]
    angle_base_last = np.arctan2(last_c_px[1] - rot_center_px[1], last_c_px[0] - rot_center_px[0])
    angle_base_t0 = angle_base_last + w_rad_s * dt_last

    last_dir_up_px = dir_up_px_list[-1]
    angle_up_offset = np.arctan2(last_dir_up_px[1], last_dir_up_px[0]) - angle_base_last

    def get_predicted_start_path_m(dt_seconds):
        """Spočítá pozici začátku tyče (+7 cm) v metrech v čase t0 + dt_seconds s predikcí 2.2s."""
        d_theta = w_rad_s * dt_seconds
        angle_future = angle_base_t0 + d_theta
        
        base_future_px = rot_center_px + radius_px * np.array([np.cos(angle_future), np.sin(angle_future)])
        angle_up_future = angle_future + angle_up_offset
        dir_up_future_px = np.array([np.cos(angle_up_future), np.sin(angle_up_future)])
        
        p0_mm = transform_pixel_to_robot(base_future_px, H)
        p_test_mm = transform_pixel_to_robot(base_future_px + dir_up_future_px * 10.0, H)
        vec_dir_mm = p_test_mm - p0_mm
        unit_dir_mm = vec_dir_mm / np.linalg.norm(vec_dir_mm)
        
        start_path_mm = p0_mm + unit_dir_mm * 70.0
        return start_path_mm / 1000.0

    def get_predicted_base_center_m(dt_seconds):
        """Spočítá pozici středu desky (base) v metrech v čase t0 + dt_seconds s predikcí 2.2s."""
        d_theta = w_rad_s * dt_seconds
        angle_future = angle_base_t0 + d_theta
        base_future_px = rot_center_px + radius_px * np.array([np.cos(angle_future), np.sin(angle_future)])
        base_future_mm = transform_pixel_to_robot(base_future_px, H)
        return base_future_mm / 1000.0

    # KROK 4: POHYB NAD ZAČÁTEK TYČE v t0 + 10s NA Z = 32 cm (PREDIKCE 2.2s)
    print("--- [KROK 4] Přesun NAD ZAČÁTEK TYČE v t0 + 10s (Z = 32 cm) s predikcí 2.2s ---")
    
    t_target_krok4 = 10.0 + ROBOT_DELAY
    pos_krok4_m = get_predicted_start_path_m(t_target_krok4)
    T_start_z32 = get_gripper_pose_for_circle_target(pos_krok4_m[0], pos_krok4_m[1], 0.32)

    time_to_wait = 10.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    if not move_bot(robot, T_start_z32):
        print("Pohyb v KROKU 4 selhal na IK. Přeskakuji na další krok...")

    # KROK 5: POHYB NA Z = 22 cm NAD ZAČÁTEK TYČE v t0 + 12s (PREDIKCE 2.2s)
    print("--- [KROK 5] Klesání NAD ZAČÁTEK TYČE v t0 + 12s (Z = 22 cm) s predikcí 2.2s ---")
    
    t_target_krok5 = 12.0 + ROBOT_DELAY
    pos_krok5_m = get_predicted_start_path_m(t_target_krok5)
    T_start_z22 = get_gripper_pose_for_circle_target(pos_krok5_m[0], pos_krok5_m[1], 0.22)

    time_to_wait = 12.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)
        
    if not move_bot(robot, T_start_z22):
        print("Pohyb v KROKU 5 selhal na IK. Přeskakuji na další krok...")

    # KROK 6: POHYB NAD STŘED BASE v t0 + 15s NA Z = 15 cm (PREDIKCE 2.2s)
    print("--- [KROK 6] Přesun NAD STŘED BASE v t0 + 15s (Z = 15 cm) s predikcí 2.2s ---")
    
    t_target_krok6 = 15.0 + ROBOT_DELAY
    pos_krok6_m = get_predicted_base_center_m(t_target_krok6)
    T_base_z15 = get_gripper_pose_for_circle_target(pos_krok6_m[0], pos_krok6_m[1], 0.15)

    time_to_wait = 15.0 - ROBOT_DELAY - (time() - t0)
    if time_to_wait > 0:
        sleep(time_to_wait)

    if not move_bot(robot, T_base_z15):
        print("Pohyb v KROKU 6 selhal na IK!")

    print("\nVšechny pohyby byly úspěšně dokončeny.")

    # Ukončení robota
    robot.stop_robot()
    robot.close_connection()