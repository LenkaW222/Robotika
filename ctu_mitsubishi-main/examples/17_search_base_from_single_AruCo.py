from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np

# ==========================================
# KONFIGURACE A MATICE HOMOGRAFIE
# ==========================================

# Vaše matice homografie H (Pixel [u,v] -> Robot [X,Y] v mm)
H = np.array([
    [-0.000192, -0.261608, 842.760080],
    [-0.267578, -0.000521, 422.583022],
    [-0.000004,  0.000005,   1.000000]
])

# ==========================================
# FUNKCE PRO DETEKCI A PŘEVOD
# ==========================================

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

def transform_pixel_to_robot(point_pixel, H_mat):
    """
    Převede bod [u, v] z pixelů na reálné [X, Y] robota v milimetrech.
    """
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H_mat @ px_homo
    x_mm = robot_homo[0] / robot_homo[2]
    y_mm = robot_homo[1] / robot_homo[2]
    return np.array([x_mm, y_mm])

# ==========================================
# HLAVNÍ LIVE BĚH
# ==========================================

if __name__ == "__main__":
    # 1. Inicializace robota a pořízení live snímku
    print("Inicializace robota a snímání z kamery...")
    robot = Rv6s(debug=False)
    robot.initialize()

    img = robot.grab_image()
    robot.close_connection()

    if img is not None:
        annotated_img = img.copy()
        
        # 2. Detekce značek a středu base
        base_center_px, markers = get_base_center_robust(img)
        
        if base_center_px is not None:
            # 3. Přepočet do souřadnic robota pomocí homografie H
            robot_xy_mm = transform_pixel_to_robot(base_center_px, H)
            print(f"Střed base v pixelech (u, v): [{base_center_px[0]:.2f}, {base_center_px[1]:.2f}] px")
            print(f"Střed base v robotu (X, Y):   [{robot_xy_mm[0]:.2f}, {robot_xy_mm[1]:.2f}] mm")
            
            # 4. Vykreslení prvků do obrázku
            # a) Vykreslení všech nalezených ArUco značek (rohy + střed)
            for m_id, pts in markers.items():
                # Vykreslení obrysu značky
                pts_int = pts.astype(np.int32)
                cv2.polylines(annotated_img, [pts_int], True, (0, 255, 255), 2)
                
                # Zvýraznění levého horního rohu (roh 0) červeně
                cv2.circle(annotated_img, tuple(pts_int[0]), 5, (0, 0, 255), -1)
                
                # Zvýraznění středu značky zeleně
                c_marker = np.mean(pts, axis=0).astype(np.int32)
                cv2.circle(annotated_img, tuple(c_marker), 5, (0, 255, 0), -1)
                cv2.putText(annotated_img, f"ID{m_id}", (c_marker[0] + 8, c_marker[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # b) Vykreslení středu base modře
            bx, by = int(base_center_px[0]), int(base_center_px[1])
            cv2.circle(annotated_img, (bx, by), 7, (255, 0, 0), -1)
            cv2.drawMarker(annotated_img, (bx, by), (255, 0, 0), cv2.MARKER_CROSS, 15, 2)
            
            # c) Textové popisy s výslednými souřadnicemi robota
            cv2.putText(annotated_img, "Stred base", (bx + 15, by - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(annotated_img, f"Robot X,Y: [{robot_xy_mm[0]:.1f}, {robot_xy_mm[1]:.1f}] mm",
                        (bx + 15, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            
            # 5. Uložení anotovaného snímku
            output_dir = "live_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "live_base_detection.jpg")
            
            cv2.imwrite(output_path, annotated_img)
            print(f"Anotovaný snímek byl úspěšně uložen do: '{output_path}'")
        else:
            print("Chyba: Nepodařilo se určit střed base.")
    else:
        print("Chyba: Z kamery nebyl získán žádný snímek.")