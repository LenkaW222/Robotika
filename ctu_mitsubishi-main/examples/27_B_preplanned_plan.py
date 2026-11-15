import cv2
import numpy as np
import os
from time import sleep, time
from ctu_mitsubishi import Rv6s

# ==========================================
# KONSTANTY A LADICÍ PARAMETRY
# ==========================================
# Časový předstih predikce (v sekundách):
# Robot dostane povel v čase 't', ale míří tam, kde tyč bude v čase 't + PREDICTION_LEAD'.
# Tím kompenzujeme dobu, než rameno fyzicky dojede do cílové polohy.
PREDICTION_LEAD = 0.2

# Matice homografie H převádí souřadnice z kamery [u, v] (px) na souřadnice robota [X, Y] (mm)
H = np.array([
    [-0.000100, -0.261713, 842.805323],
    [-0.267645, -0.000507, 422.676470],
    [-0.000004,  0.000005,   1.000000]
])

# Bezpečná výchozí (domovská) poloha robota v metrech
Home_position = np.array([
    [1,  0,  0,  0.4],
    [0, -1,  0, -0.3],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]
])


# ==========================================
# POMOCNÉ FUNKCE PRO KINEMATIKU A VIDĚNÍ
# ==========================================
def transform_pixel_to_robot(point_pixel, H_mat):
    """Převede 2D bod [u, v] z obrazu (v pixelech) na reálné [X, Y] robota v milimetrech."""
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])


def get_ik_nearest(robot, position, q_ref):
    """
    Spočítá inverzní kinematiku (IK) pro cílovou matici 'position'.
    Vybere konfiguraci kloubů nejbližší předchozí poloze 'q_ref'.
    Vrací None, pokud řešení neexistuje.
    """
    if position is None:
        return None
    qs = robot.ik(position)
    if qs is None or len(qs) == 0:
        return None
    return min(qs, key=lambda q: np.linalg.norm(q - q_ref))


def move_bot(robot, position):
    """Okamžitý blokující pohyb na zadanou pozici (pro Home a parkování nad střed)."""
    q0 = robot.get_q()
    qnew = get_ik_nearest(robot, position, q0)
    if qnew is None:
        print(f"[CHYBA IK] Pro pozici X={position[0,3]:.3f}, Y={position[1,3]:.3f}, Z={position[2,3]:.3f} neexistuje řešení.")
        return False
    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)
    return True


def move_home(robot):
    """Bezpečný návrat do Home (nejprve zvedne nástroj nad Z = 32 cm, aby nenarazil do tyče)."""
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
    """
    Spočítá cílovou transformační matici příruby gripperu (4x4) tak, aby se STŘED OBRUČE
    nacházel přesně v souřadnicích [x_target_m, y_target_m, z_target_m].
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


def get_base_info_px(img):
    """Detekuje ArUco značky (ID 1 a ID 2) na základně a vrátí střed base a směrový vektor UP."""
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
    """Proloží nasbírané body středu base kružnicí a najde střed rotace."""
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = -a / 2.0, -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    return np.array([xc, yc]), radius


def calculate_rotation(points, timestamps, center_pt):
    """Spočítá úhlovou rychlost otáčení stolu w (rad/s) a periodu jedné otáčky."""
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
# HLAVNÍ PROGRAM
# ==========================================
if __name__ == "__main__":
    robot = Rv6s(debug=False)
    robot.initialize()

    # ---------------------------------------------------------
    # FÁZE 1 (t = 0 s): Přesun robota do Home pozice
    # ---------------------------------------------------------
    if not move_home(robot):
        print("Chyba při startu do Home position. Ukončuji program.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # ---------------------------------------------------------
    # FÁZE 2 (t = 0 s až 20 s): Snímání rotace a identifikace parametrů
    # ---------------------------------------------------------
    base_centers_px = []
    timestamps = []
    dir_up_px_list = []
    
    total_duration = 20.0
    interval = 1.0
    
    print("\n--- FÁZE 2: Snímání otáčení stolku (20s) ---")
    t_start_scan = time()
    last_grab = -interval

    while (time() - t_start_scan) < total_duration:
        curr_t = time() - t_start_scan
        if curr_t - last_grab >= interval:
            last_grab = curr_t
            img = robot.grab_image()
            if img is not None:
                c_px, d_up_px = get_base_info_px(img)
                if c_px is not None and d_up_px is not None:
                    d_up_px = -d_up_px
                    base_centers_px.append(c_px)
                    timestamps.append(curr_t)
                    dir_up_px_list.append(d_up_px)
                    print(f"[{curr_t:.1f}s] Detekován střed base v px: ({c_px[0]:.1f}, {c_px[1]:.1f})")
        sleep(0.05)

    if len(base_centers_px) < 4:
        print("Chyba: Nedostatek bodů pro analýzu rotace.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    rot_center_px, radius_px = fit_circle_least_squares(base_centers_px)
    w_rad_s, speed_deg_s, period_s = calculate_rotation(base_centers_px, timestamps, rot_center_px)
    
    rot_center_mm = transform_pixel_to_robot(rot_center_px, H)
    rot_center_m = rot_center_mm / 1000.0
    print(f"\nStřed rotace: X={rot_center_m[0]:.4f} m, Y={rot_center_m[1]:.4f} m | Perioda: {period_s:.2f} s ({speed_deg_s:.2f} deg/s)")

    t0 = time()
    dt_last = (t0 - t_start_scan) - timestamps[-1]

    last_c_px = base_centers_px[-1]
    angle_base_last = np.arctan2(last_c_px[1] - rot_center_px[1], last_c_px[0] - rot_center_px[0])
    angle_base_t0 = angle_base_last + w_rad_s * dt_last

    last_dir_up_px = dir_up_px_list[-1]
    angle_up_offset = np.arctan2(last_dir_up_px[1], last_dir_up_px[0]) - angle_base_last

    def predict_base_and_start_m(dt_from_t0):
        """
        Předpoví polohu středu base a začátku tyče B v čase (t0 + dt_from_t0).
        """
        angle_future = angle_base_t0 + w_rad_s * dt_from_t0
        base_future_px = rot_center_px + radius_px * np.array([np.cos(angle_future), np.sin(angle_future)])
        
        angle_up_future = angle_future + angle_up_offset
        dir_up_future_px = np.array([np.cos(angle_up_future), np.sin(angle_up_future)])
        
        p_base_mm = transform_pixel_to_robot(base_future_px, H)
        p_test_mm = transform_pixel_to_robot(base_future_px + dir_up_future_px * 10.0, H)
        
        vec_dir_mm = p_test_mm - p_base_mm
        unit_dir_mm = vec_dir_mm / np.linalg.norm(vec_dir_mm)
        p_start_mm = p_base_mm + unit_dir_mm * 70.0
        
        return p_base_mm / 1000.0, p_start_mm / 1000.0

    # ---------------------------------------------------------
    # FÁZE 3: Přesun obruče nad střed otáčení (Z = 32 cm)
    # ---------------------------------------------------------
    print("\n--- FÁZE 3: Přesun středu obruče nad střed otáčení (Z = 32 cm) ---")
    T_above_rot_center = get_gripper_pose_for_circle_target(rot_center_m[0], rot_center_m[1], 0.32)
    if not move_bot(robot, T_above_rot_center):
        print("Nepodařilo se najet nad střed rotace.")
        robot.stop_robot()
        robot.close_connection()
        exit()

    # ---------------------------------------------------------
    # FÁZE 4: Předvýpočet celé trajektorie do Qplan (s kompenzací PREDICTION_LEAD)
    # ---------------------------------------------------------
    print(f"\n--- FÁZE 4: Výpočet trajektorie (předstih predikce PREDICTION_LEAD = {PREDICTION_LEAD} s) ---")
    q_parked = robot.get_q()
    t_start_candidate = max(40.0 - (t0 - t_start_scan), (time() - t0) + 2.0)

    Qplan = []
    valid_plan_found = False

    while not valid_plan_found:
        global_start_t = (t0 - t_start_scan) + t_start_candidate
        print(f"Počítám IK pro spuštění v t = {global_start_t:.2f} s (cílové polohy pro t + {PREDICTION_LEAD} s)...")
        
        temp_plan = []
        q_prev = q_parked
        ik_failed = False

        # Každý prvek je (cas_odeslani_prikazu, x_m, y_m, z_m, popis)
        # Pozice tyče se ale vždy počítá pro čas (cas_odeslani_prikazu + PREDICTION_LEAD)
        waypoints = []

        # POHYB 1 (odešle se v t = 40.0 s, cílí na polohu tyče v t = 40.0 + PREDICTION_LEAD)
        t1_cmd = t_start_candidate
        _, start_1_m = predict_base_and_start_m(t1_cmd + PREDICTION_LEAD)
        waypoints.append((t1_cmd, start_1_m[0], start_1_m[1], 0.30, f"1. Nad začátek tyče (Z = 30 cm, predikce pro +{PREDICTION_LEAD}s)"))

        # POHYB 2 (odešle se v t = 41.0 s, cílí na polohu tyče v t = 41.0 + PREDICTION_LEAD)
        t2_cmd = t_start_candidate + 1.0
        _, start_2_m = predict_base_and_start_m(t2_cmd + PREDICTION_LEAD)
        waypoints.append((t2_cmd, start_2_m[0], start_2_m[1], 0.22, f"2. Sjezd na Z = 22 cm (predikce pro +{PREDICTION_LEAD}s)"))

        # MEZIKROK (odešle se v t = 41.5 s, cílí na polovinu šikmé části v t = 41.5 + PREDICTION_LEAD)
        t_mid_cmd = t_start_candidate + 1.5
        base_mid_m, start_mid_m = predict_base_and_start_m(t_mid_cmd + PREDICTION_LEAD)
        x_mid = 0.5 * (start_mid_m[0] + base_mid_m[0])
        y_mid = 0.5 * (start_mid_m[1] + base_mid_m[1])
        waypoints.append((t_mid_cmd, x_mid, y_mid, 0.185, f"2b. Mezikrok v polovině šikmé části (Z = 18.5 cm, predikce pro +{PREDICTION_LEAD}s)"))

        # POHYB 3 (odešle se v t = 42.0 s, cílí na střed base v t = 42.0 + PREDICTION_LEAD)
        t3_cmd = t_start_candidate + 2.0
        base_3_m, _ = predict_base_and_start_m(t3_cmd + PREDICTION_LEAD)
        waypoints.append((t3_cmd, base_3_m[0], base_3_m[1], 0.15, f"3. Nad střed base (Z = 15 cm, predikce pro +{PREDICTION_LEAD}s)"))

        # POHYB 4 (odešle se v t = 43.0 s, cílí na střed base v t = 43.0 + PREDICTION_LEAD)
        t4_cmd = t_start_candidate + 3.0
        base_4_m, _ = predict_base_and_start_m(t4_cmd + PREDICTION_LEAD)
        waypoints.append((t4_cmd, base_4_m[0], base_4_m[1], 0.095, f"4. Cílová pozice u base (Z = 9.5 cm, predikce pro +{PREDICTION_LEAD}s)"))

        # Výpočet inverzní kinematiky (IK) pro všechny body
        for (t_exec, x_m, y_m, z_m, desc) in waypoints:
            T_target = get_gripper_pose_for_circle_target(x_m, y_m, z_m)
            q_next = get_ik_nearest(robot, T_target, q_prev)
            
            if q_next is None:
                print(f"  [IK SELHALO] u bodu '{desc}'. Posouvám start o +1.0 s a počítám znovu.")
                ik_failed = True
                break
                
            temp_plan.append((t_exec, q_next, desc))
            q_prev = q_next

        if not ik_failed:
            if (time() - t0) > temp_plan[0][0] - 0.5:
                print("  [ČAS VYPRŠEL] Výpočet trval moc dlouho. Posouvám start o +2.0 s.")
                t_start_candidate += 2.0
            else:
                Qplan = temp_plan
                valid_plan_found = True
                print(f"Kompletní trasa uložena do Qplan ({len(Qplan)} bodů)!")
        else:
            t_start_candidate += 1.0

    # ---------------------------------------------------------
    # FÁZE 5: Časově řízené vykonání předpočítané trasy
    # ---------------------------------------------------------
    print("\n--- FÁZE 5: Čekání na spuštění a vykonání pohybu ---")
    idx = 0
    while idx < len(Qplan):
        elapsed_from_t0 = time() - t0
        target_t, q_cmd, desc = Qplan[idx]
        
        if elapsed_from_t0 >= target_t:
            real_time_total = time() - t_start_scan
            print(f"[t = {real_time_total:.2f}s] {desc}")
            robot.move_to_q(q_cmd)
            idx += 1
        else:
            sleep(0.005)

    robot.wait_for_motion_stop(poll_interval=0.01)
    print("\nPrůchod úspěšně dokončen!")

    robot.stop_robot()
    robot.close_connection()