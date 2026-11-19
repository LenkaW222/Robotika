import cv2
import numpy as np
import os
from time import sleep, time
from ctu_mitsubishi import Rv6s

# ==========================================
# KONSTANTY A LADICÍ PARAMETRY
# ==========================================
# Časový předstih predikce (v sekundách) pro kompenzaci doby pohybu ramene
PREDICTION_LEAD = 1.6

# Matice homografie H převádí souřadnice z kamery [u, v] (px) na souřadnice robota [X, Y] (mm)
H = np.array([
    [-0.000100, -0.261713, 842.805323],
    [-0.267645, -0.000507, 422.676470],
    [-0.000004,  0.000005,   1.000000]
])

# Bezpečná výchozí (domovská) poloha robota v metrech[cite: 3, 4]
Home_position = np.array([
    [1,  0,  0,  0.4],
    [0, -1,  0, -0.3],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]
])


# ==========================================
# POMOCNÉ FUNKCE PRO KINEMATIKU A VIDĚNÍ
# ==========================================
def safe_exit(robot, message):
    """Vypíše chybovou zprávu, bezpečně zastaví robota a ukončí program."""
    print(f"\n[UKONČUJI PROGRAM] z duvodu: {message}")
    robot.stop_robot()
    robot.close_connection()
    exit()


def transform_pixel_to_robot(point_pixel, H_mat):
    """Převede 2D bod [u, v] z obrazu (v pixelech) na reálné [X, Y] robota v milimetrech[cite: 3, 4]."""
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])


def get_ik_nearest(robot, position, q_ref):
    """
    Spočítá inverzní kinematiku (IK) pro cílovou matici 'position'[cite: 3, 4].
    Vybere konfiguraci kloubů nejbližší předchozí poloze 'q_ref'[cite: 3, 4].
    Vrací None, pokud řešení neexistuje[cite: 4].
    """
    if position is None:
        return None
    qs = robot.ik(position)
    if qs is None or len(qs) == 0:
        return None
    return min(qs, key=lambda q: np.linalg.norm(q - q_ref))


def check_trajectory_collision(robot, q_start, q_end, steps=10):
    """
    Zkontroluje přechod mezi kloubovými konfiguracemi q_start a q_end.
    Vrací (True, duvod_kolize) pokud hrozí kolize, jinak (False, "").
    """
    # 1. Kontrola skokového přetočení kloubů (ochrana proti singularitě a švihnutí ramene)
    # max_joint_diff = np.max(np.abs(q_end - q_start))
    # if max_joint_diff > 1.5:  # cca 86 stupňů v jednom kroku
    #     return True, f"Nebezpečný skok kloubu o {np.degrees(max_joint_diff):.1f}° (riziko kolize při přetočení)."

    # # 2. Kontrola mezibodů pomocí dopředné kinematiky (FK)
    for i in range(1, steps + 1):
        alpha = i / steps
        q_interp = (1.0 - alpha) * q_start + alpha * q_end
        T_fk = robot.fk(q_interp)
        
        p_gripper = T_fk[0:3, 3]
        R_gripper = T_fk[0:3, 0:3]
        p_circle = p_gripper + R_gripper @ np.array([0.135, 0.0, 0.030])

        # A) Kolize se základnou / stolem (base leží v Z = 7.0 cm)
        if p_circle[2] < 0.085 or p_gripper[2] < 0.115:
            return True, f"Nízká výška nad stolem (Z_obruč={p_circle[2]*100:.1f} cm, Z_gripper={p_gripper[2]*100:.1f} cm)."

        # B) Autokolize: zápěstí příliš blízko ke sloupu / základně robota v rovině XY
        dist_from_base_xy = np.hypot(p_gripper[0], p_gripper[1])
        if dist_from_base_xy < 0.22:
            return True, f"Gripper příliš blízko tělu robota (r={dist_from_base_xy*100:.1f} cm < 22 cm)."

    return False, ""


def move_bot(robot, position):
    """Okamžitý blokující pohyb na zadanou pozici (pro Home a parkování nad střed)[cite: 3, 4]."""
    q0 = robot.get_q()
    qnew = get_ik_nearest(robot, position, q0)
    if qnew is None:
        print(f"[CHYBA IK] Pro pozici X={position[0,3]:.3f}, Y={position[1,3]:.3f}, Z={position[2,3]:.3f} neexistuje řešení[cite: 4].")
        return False
    
    collision, reason = check_trajectory_collision(robot, q0, qnew)
    if collision:
        safe_exit(robot, f"Hrozí kolize při přípravném pohybu: {reason}")

    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)
    return True


def move_home(robot):
    """Bezpečný návrat do Home (nejprve zvedne nástroj nad Z = 32 cm, aby nenarazil do tyče)[cite: 3, 4]."""
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)
    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift[2, 3] = 0.32
        if not move_bot(robot, T_lift):
            return False
    return move_bot(robot, Home_position)


def get_gripper_pose_for_circle_target(x_target_m, y_target_m, z_target_m, rotate_90_xy=False):
    """
    Spočítá cílovou transformační matici příruby gripperu (4x4) tak, aby se STŘED OBRUČE
    nacházel přesně v souřadnicích [x_target_m, y_target_m, z_target_m][cite: 3, 4].
    Pokud je rotate_90_xy=True (když X středu otáčení < 40 cm), otočí orientaci gripperu
    o 90 stupňů v rovině XY, aby robot nenarazil sám do sebe.
    """
    if rotate_90_xy:
        # Otočení o 90° kolem svislé osy Z v rovině XY
        R_gripper = np.array([
            [0,  1,  0],
            [1,  0,  0],
            [0,  0, -1]
        ])
    else:
        # Standardní orientace gripperu[cite: 3, 4]
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
    """Detekuje ArUco značky (ID 1 a ID 2) na základně a vrátí střed base a směrový vektor UP[cite: 3, 4]."""
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
    """Proloží nasbírané body středu base kružnicí a najde střed rotace[cite: 4]."""
    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    xc, yc = -a / 2.0, -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    return np.array([xc, yc]), radius


def calculate_rotation(points, timestamps, center_pt):
    """Spočítá úhlovou rychlost otáčení stolu w (rad/s) a periodu jedné otáčky[cite: 4]."""
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
        safe_exit(robot, "Chyba při přesunu do Home position.")

    # ---------------------------------------------------------
    # FÁZE 2 (t = 0 s až 20 s): Snímání rotace a identifikace parametrů
    # ---------------------------------------------------------
    base_centers_px = []
    timestamps = []
    dir_up_px_list = []
    
    total_duration = 20.0
    interval = 1.0
    total_frames = 0
    
    print("\n--- FÁZE 2: Snímání otáčení stolku (20s) ---")
    t_start_scan = time()
    last_grab = -interval

    while (time() - t_start_scan) < total_duration:
        curr_t = time() - t_start_scan
        if curr_t - last_grab >= interval:
            last_grab = curr_t
            img = robot.grab_image()
            if img is not None:
                total_frames += 1
                c_px, d_up_px = get_base_info_px(img)
                if c_px is not None and d_up_px is not None:
                    d_up_px = -d_up_px
                    base_centers_px.append(c_px)
                    timestamps.append(curr_t)
                    dir_up_px_list.append(d_up_px)
                    print(f"[{curr_t:.1f}s] Detekováno ID -> střed base v px: ({c_px[0]:.1f}, {c_px[1]:.1f})")
                else:
                    print(f"[{curr_t:.1f}s] VAROVÁNÍ: Nebyly detekovány žádné ArUco značky (IDs)!")
        sleep(0.05)

    # KONTROLA ÚSPĚŠNOSTI DETEKCE ArUco ZNAČEK
    detected_count = len(base_centers_px)
    if detected_count == 0:
        safe_exit(robot, "Během 20s snímání nebyly vidět ŽÁDNÉ ArUco značky (IDs).")

    if total_frames == 0 or detected_count <= (total_frames / 2.0):
        safe_exit(
            robot,
            f"Nedostatečná detekce ArUco značek: viděno pouze v {detected_count} z {total_frames} snímků "
            f"({(detected_count / max(1, total_frames)) * 100:.1f} % <= 50 %)."
        )

    print(f"\nÚspěšnost detekce značek: {detected_count}/{total_frames} snímků ({(detected_count/total_frames)*100:.1f} %).")

    # Výpočet středu rotace a rychlosti
    rot_center_px, radius_px = fit_circle_least_squares(base_centers_px)
    w_rad_s, speed_deg_s, period_s = calculate_rotation(base_centers_px, timestamps, rot_center_px)

    rot_center_mm = transform_pixel_to_robot(rot_center_px, H)
    rot_center_m = rot_center_mm / 1000.0
    
    y_min = -17
    y_max = 36
    x_min = 37
    x_max = 80
    max_dist = 80

    dist = np.sqrt((rot_center_m[0]*100)**2 + (rot_center_m[1]*100)**2)

    if rot_center_m[0]*100 < x_min or rot_center_m[0]*100 > x_max or rot_center_m[1]*100 < y_min or rot_center_m[1]*100 > y_max or dist > max_dist:
        safe_exit(robot,f"Stred rotace bludiste je mimo pracovni prostor robota: X={rot_center_m[0]*100:.1f} cm, Y={rot_center_m[1]*100:.1f} cm, vzdalenost: {dist:.1f} cm")

    print(f"Střed rotace: X={rot_center_m[0]*100:.1f} cm, Y={rot_center_m[1]*100:.1f} cm | Perioda: {period_s:.2f} s ({speed_deg_s:.2f} deg/s)")
    print(f"Vzdalenost stredu otaceni od poatku robota: {dist:.1f} cm")
    # ROZHODNUTÍ O OTOČENÍ ROTAČNÍ MATICE GRIPPERU O 90° V XY (pokud X < 40 cm)
    use_rotated_gripper = rot_center_m[0] < 0.40
    if use_rotated_gripper:
        print(f"[INFO] X středu otáčení ({rot_center_m[0]*100:.1f} cm) < 40 cm -> Otáčím matici gripperu o 90° v rovině XY proti autokolizi.")
    else:
        print(f"[INFO] X středu otáčení ({rot_center_m[0]*100:.1f} cm) >= 40 cm -> Používám standardní orientaci gripperu.")

    t0 = time()
    dt_last = (t0 - t_start_scan) - timestamps[-1]

    last_c_px = base_centers_px[-1]
    angle_base_last = np.arctan2(last_c_px[1] - rot_center_px[1], last_c_px[0] - rot_center_px[0])
    angle_base_t0 = angle_base_last + w_rad_s * dt_last

    last_dir_up_px = dir_up_px_list[-1]
    angle_up_offset = np.arctan2(last_dir_up_px[1], last_dir_up_px[0]) - angle_base_last

    def predict_base_and_start_m(dt_from_t0):
        """Předpoví polohu středu base a začátku tyče B v čase (t0 + dt_from_t0)[cite: 4]."""
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
    T_above_rot_center = get_gripper_pose_for_circle_target(
        rot_center_m[0], rot_center_m[1], 0.32, rotate_90_xy=use_rotated_gripper
    )
    if not move_bot(robot, T_above_rot_center):
        print("Nejede nad stred rot")
        #safe_exit(robot, "Nepodařilo se najet nad střed rotace (neexistuje IK).")

    # ---------------------------------------------------------
    # FÁZE 4: Předvýpočet celé trajektorie do Qplan + KONTROLA KOLIZÍ
    # ---------------------------------------------------------
    print(f"\n--- FÁZE 4: Výpočet trajektorie (předstih predikce PREDICTION_LEAD = {PREDICTION_LEAD} s) ---")
    q_parked = robot.get_q()
    t_start_candidate = max(40.0 - (t0 - t_start_scan), (time() - t0) + 2.0)

    Qplan = []
    valid_plan_found = False

    while not valid_plan_found:
        global_start_t = (t0 - t_start_scan) + t_start_candidate
        if global_start_t > 100:
            safe_exit(robot, f" Nedosazitelna poloha stolku: x: {rot_center_m[0]:.4f}, y: {rot_center_m[1]:.4f}")
        print(f"Počítám IK pro spuštění v t = {global_start_t:.2f} s (predikce pro t + {PREDICTION_LEAD} s)...")
        
        temp_plan = []
        q_prev = q_parked
        ik_failed = False

        waypoints = []

        # POHYB 1 (t = 40.0 s): Nájezd nad začátek tyče ve výšce Z = 30 cm
        t1_cmd = t_start_candidate
        _, start_1_m = predict_base_and_start_m(t1_cmd + PREDICTION_LEAD)
        waypoints.append((t1_cmd, start_1_m[0], start_1_m[1], 0.30, "1. Nad začátek tyče (Z = 30 cm)"))

        # POHYB 2 (t = 41.0 s): Svislý sjezd na Z = 22 cm nad začátek tyče
        t2_cmd = t_start_candidate + 1.0
        _, start_2_m = predict_base_and_start_m(t2_cmd + PREDICTION_LEAD)
        waypoints.append((t2_cmd, start_2_m[0], start_2_m[1], 0.22, "2. Sjezd na Z = 22 cm"))

        # MEZIKROK (t = 41.5 s): Polovina šikmé části tyče v Z = 18.5 cm
        t_mid_cmd = t_start_candidate + 1.5
        base_mid_m, start_mid_m = predict_base_and_start_m(t_mid_cmd + PREDICTION_LEAD)
        x_mid = 0.5 * (start_mid_m[0] + base_mid_m[0])
        y_mid = 0.5 * (start_mid_m[1] + base_mid_m[1])
        waypoints.append((t_mid_cmd, x_mid, y_mid, 0.185, "2b. Mezikrok v polovině šikmé části (Z = 18.5 cm)"))

        # POHYB 3 (t = 42.0 s): Konec šikmé části v Z = 15 cm přímo nad středem base
        t3_cmd = t_start_candidate + 2.0
        base_3_m, _ = predict_base_and_start_m(t3_cmd + PREDICTION_LEAD)
        waypoints.append((t3_cmd, base_3_m[0], base_3_m[1], 0.15, "3. Nad střed base (Z = 15 cm)"))

        # POHYB 4 (t = 43.0 s): Svislý sjezd dolů na cílovou výšku Z = 9.5 cm nad středem base
        t4_cmd = t_start_candidate + 3.0
        base_4_m, _ = predict_base_and_start_m(t4_cmd + PREDICTION_LEAD)
        waypoints.append((t4_cmd, base_4_m[0], base_4_m[1], 0.095, "4. Cílová pozice u base (Z = 9.5 cm)"))

        # Výpočet IK a kontrola kolizí pro všechny body v trajektorii
        for (t_exec, x_m, y_m, z_m, desc) in waypoints:
            T_target = get_gripper_pose_for_circle_target(
                x_m, y_m, z_m, rotate_90_xy=use_rotated_gripper
            )
            q_next = get_ik_nearest(robot, T_target, q_prev)
            
            if q_next is None:
                print(f"  [IK SELHALO] u bodu '{desc}'. Posouvám start o +1.0 s a počítám znovu.")
                ik_failed = True
                break

            # KONTROLA KOLIZE PRO NALEZENOU TRASU
            collision, reason = check_trajectory_collision(robot, q_prev, q_next)
            if collision:
                safe_exit(robot, f"V nalezené trase hrozí kolize u bodu '{desc}': {reason}")
                
            temp_plan.append((t_exec, q_next, desc))
            q_prev = q_next

        if not ik_failed:
            if (time() - t0) > temp_plan[0][0] - 0.5:
                print("  [ČAS VYPRŠEL] Výpočet trval moc dlouho. Posouvám start o +2.0 s.")
                t_start_candidate += 2.0
            else:
                Qplan = temp_plan
                valid_plan_found = True
                print(f"Kompletní bezkolizní trasa uložena do Qplan ({len(Qplan)} bodů)!")
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