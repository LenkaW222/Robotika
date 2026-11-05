from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np
from time import sleep, time

# ==========================================
# POMOCNÉ FUNKCE PRO DETEKCI A GEOMETRII
# ==========================================

def get_base_center_px(img):
    """
    Detekuje ArUco značky ID1 a ID2 a spočítá jejich střed v pixelech.
    """
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
        return (centers[1] + centers[2]) / 2.0
    return None

def fit_circle_least_squares(points):
    """
    Proloží množinu bodů (x, y) nejlepší kružnicí pomocí metody nejmenších čtverců.
    Vrací: (xc, yc), R
    """
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Sestavení matice pro lineární nejmenší čtverce: A * [a, b, c]^T = B
    # Rovnice kružnice: (x - xc)^2 + (y - yc)^2 = R^2  =>  x^2 + y^2 + a*x + b*y + c = 0
    A = np.column_stack([x, y, np.ones_like(x)])
    B = -(x**2 + y**2)
    
    # Řešení nejmenších čtverců
    a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
    
    # Přepočet na střed xc, yc a poloměr R
    xc = -a / 2.0
    yc = -b / 2.0
    radius = np.sqrt(xc**2 + yc**2 - c)
    
    return np.array([xc, yc]), radius

def calculate_rotation_speed_and_direction(points, timestamps, center_pt):
    """
    Spočítá rychlost (v ot/min a deg/s) a směr rotace (CW/CCW) z časové řady bodů.
    """
    pts = np.array(points)
    xc, yc = center_pt
    
    # Výpočet polárních úhlů jednotlivých bodů vůči středu rotace
    angles = np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc)
    
    # Rozbalení úhlů (unwrap), aby se předešlo skokům při přechodu přes +/- pi
    angles_unwrapped = np.unwrap(angles)
    
    # Změny úhlů a časů
    delta_angles = np.diff(angles_unwrapped)
    delta_times = np.diff(timestamps)
    
    # Úhlová rychlost v rad/s
    angular_velocities = delta_angles / delta_times
    mean_w_rad_s = np.mean(angular_velocities)
    
    # Směr rotace (v souřadnicovém systému obrázku s osou Y dolů)
    # Kladný přírůstek v arctan2 při osách (X doprava, Y dolů) znamená hodiny (CW)
    direction = "Ve směru hodinových ručiček (CW)" if mean_w_rad_s > 0 else "Proti směru hodinových ručiček (CCW)"
    
    speed_deg_s = abs(np.degrees(mean_w_rad_s))
    speed_rpm = (abs(mean_w_rad_s) * 60.0) / (2 * np.pi)
    period_sec = (2 * np.pi) / abs(mean_w_rad_s) if mean_w_rad_s != 0 else 0
    
    return speed_deg_s, speed_rpm, period_sec, direction


# ==========================================
# MAIN SKRIPT
# ==========================================

if __name__ == "__main__":
    # Inicializace robota / kamery
    robot = Rv6s(debug=False)
    robot.initialize()

    base_centers_px = []
    timestamps = []

    total_duration = 15.0  # Snos 15 sekund pro bezpečné pokrytí 1 celé rotace (13s)
    interval = 1.0        # Snímání každou 1 sekundu

    print(f"Začíná snímání rotačního stolku po dobu {total_duration}s...")
    start_time = time()
    last_grab_time = 0

    captured_images = []

    # 1. Snímání sekvence obrázků
    while (time() - start_time) < total_duration:
        current_time = time() - start_time
        
        if current_time - last_grab_time >= interval:
            last_grab_time = current_time
            
            img = robot.grab_image()
            if img is not None:
                center_px = get_base_center_px(img)
                
                if center_px is not None:
                    base_centers_px.append(center_px)
                    timestamps.append(current_time)
                    captured_images.append(img.copy())
                    print(f"[{current_time:.2f}s] Detekován střed base na (u, v): [{center_px[0]:.1f}, {center_px[1]:.1f}]")
                else:
                    print(f"[{current_time:.2f}s] Base nebyla detekována.")
                    
        sleep(0.05)

    robot.close_connection()

    # 2. Vyhodnocení geometrie a rotace
    if len(base_centers_px) >= 4:
        # Metoda nejmenších čtverců pro proložení kružnice
        rotation_center_px, radius_px = fit_circle_least_squares(base_centers_px)
        
        # Rychlost a směr
        speed_deg, speed_rpm, period_s, direction = calculate_rotation_speed_and_direction(
            base_centers_px, timestamps, rotation_center_px
        )
        
        print("\n" + "="*50)
        print(" VÝSLEDKY ANALÝZY ROTAČNÍHO STOLKU")
        print("="*50)
        print(f"Střed rotace stolku (px):   [{rotation_center_px[0]:.2f}, {rotation_center_px[1]:.2f}]")
        print(f"Vzdálenost středu base od středu rotace (poloměr): {radius_px:.2f} px")
        print(f"Směr rotace:                {direction}")
        print(f"Rychlost rotace:            {speed_deg:.2f} °/s ({speed_rpm:.2f} ot/min)")
        print(f"Odhad doby 1 otáčky:        {period_s:.2f} s")
        print("="*50)
        
        # 3. Vykreslení výsledků do posledního zachyceného snímku
        last_img = captured_images[-1]
        annotated_img = last_img.copy()
        
        # Vykreslení trajektorie (všech nalezených středy base)
        for pt in base_centers_px:
            cv2.circle(annotated_img, (int(pt[0]), int(pt[1])), 4, (255, 255, 0), -1)
            
        # Vykreslení proložené kružnice rotace
        center_tuple = (int(rotation_center_px[0]), int(rotation_center_px[1]))
        cv2.circle(annotated_img, center_tuple, int(radius_px), (0, 255, 0), 2)
        
        # Vykreslení středu rotace (křížek)
        cv2.drawMarker(annotated_img, center_tuple, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(annotated_img, "Stred rotace stolku", (center_tuple[0] + 15, center_tuple[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        # Spojnice středu rotace a aktuální polohy base
        last_base_pt = (int(base_centers_px[-1][0]), int(base_centers_px[-1][1]))
        cv2.line(annotated_img, center_tuple, last_base_pt, (255, 0, 255), 2)
        cv2.circle(annotated_img, last_base_pt, 6, (255, 0, 0), -1)
        
        # Vypis textových informací do obrázku
        info_text1 = f"Rychlost: {speed_deg:.2f} deg/s ({speed_rpm:.2f} RPM)"
        info_text2 = f"Perioda: {period_s:.2f} s/ot | Polomer: {radius_px:.1f} px"
        info_text3 = f"Smer: {direction}"
        
        cv2.putText(annotated_img, info_text1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_img, info_text2, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_img, info_text3, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Uložení výsledného snímku
        output_path = "rotation_analysis_result.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\nVýsledný snímek byl uložen do: {output_path}")

    else:
        print("K dispozici nebylo dostatek bodů pro proložení kružnice (vyžadováno min. 4 body).")