import os
import cv2
import numpy as np
from ctu_mitsubishi import Rv6s


def find_hoop_center_cv(img):
    """
    Detekuje kruh v obrázku pomocí Houghovy transformace
    a vrátí souřadnice středu (u, v) v pixelech a poloměr.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detekce kružnice (parametry přizpůsobeny velikosti kruhu na snímku)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=80
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return float(circles[0][0]), float(circles[0][1]), float(circles[0][2])
    
    return None, None, None


def compute_homography_and_find_centers(img):
    """
    Vypočítá homografii H z ArUco značek a převede detekovaný střed kruhu
    z pixelů na 2D souřadnice desky v milimetrech.
    """
    # 1. Detekce ArUco značek
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        raise ValueError("V obrázku nebyly detekovány žádné ArUco značky.")

    # 2. Lokální souřadnice rohů značek 40x40 mm na desce 125x125 mm
    marker_square = np.array([
        [0.0, 0.0],
        [40.0, 0.0],
        [40.0, 40.0],
        [0.0, 40.0]
    ], dtype=np.float32)

    marker_offsets = {
        1: (17.5, 17.5),
        2: (80.0, 80.0)
    }

    world_pts = []
    image_pts = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_offsets:
            offset = np.array(marker_offsets[marker_id], dtype=np.float32)
            pts_world = marker_square + offset
            pts_img = corners[i].reshape(4, 2)

            world_pts.append(pts_world)
            image_pts.append(pts_img)

    if not world_pts:
        raise ValueError("Nebyly nalezeny požadované ArUco značky (ID 1 ani 2).")

    world_pts = np.concatenate(world_pts, axis=0).astype(np.float32)
    image_pts = np.concatenate(image_pts, axis=0).astype(np.float32)

    # 3. Výpočet homografie H a její inverze H_inv
    H, _ = cv2.findHomography(world_pts, image_pts)
    H_inv = np.linalg.inv(H)

    # 4. Projekce středu desky (62.5, 62.5 mm) do pixelů
    board_center_mm = np.array([62.5, 62.5, 1.0])
    board_px_homog = H @ board_center_mm
    board_pixel = board_px_homog[:2] / board_px_homog[2]

    # 5. Detekce středu kruhu přímo z obrázku
    hoop_u, hoop_v, hoop_radius = find_hoop_center_cv(img)

    if hoop_u is not None:
        hoop_pixel_homog = np.array([hoop_u, hoop_v, 1.0])
        hoop_board_mm = H_inv @ hoop_pixel_homog
        hoop_board_mm = hoop_board_mm[:2] / hoop_board_mm[2]

        print(f"Kruh detekován na pixelech: (u={hoop_u:.1f}, v={hoop_v:.1f})")
        print(f"Pozice kruhu na desce: X={hoop_board_mm[0]:.1f} mm, Y={hoop_board_mm[1]:.1f} mm")
        hoop_pixel = np.array([hoop_u, hoop_v])
    else:
        print("Varování: Kruh se nepodařilo v obrázku detekovat.")
        hoop_pixel = None

    return H, board_pixel, hoop_pixel, hoop_radius


# --- Hlavní kód bez bloku try ---

# 1. Inicializace robota
robot = Rv6s(debug=False)
robot.initialize()

# Pořízení snímku z kamery ze současné pozice
img = robot.grab_image()

# Ukončení připojení k robotu
robot.stop_robot()
robot.close_connection()

if img is not None:
    # 2. Vytvoření složky photos_homo a uložení originálního snímku
    output_dir = "photos_homo"
    os.makedirs(output_dir, exist_ok=True)

    orig_path = os.path.join(output_dir, "original_image.png")
    cv2.imwrite(orig_path, img)
    print(f"Originální snímek byl uložen do: {orig_path}")

    # 3. Výpočet homografie a nalezení středů
    H, board_px, hoop_px, r = compute_homography_and_find_centers(img)

    # 4. Vykreslení středů do snímku
    img_annotated = img.copy()

    # Zelený křížek = Střed základní desky
    b_center = (int(round(board_px[0])), int(round(board_px[1])))
    cv2.drawMarker(img_annotated, b_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img_annotated, "Base Center", (b_center[0] + 10, b_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Červený křížek + Kružnice = Detekovaný střed kruhu
    if hoop_px is not None:
        h_center = (int(round(hoop_px[0])), int(round(hoop_px[1])))
        cv2.drawMarker(img_annotated, h_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if r is not None:
            cv2.circle(img_annotated, h_center, int(r), (0, 0, 255), 2)
        cv2.putText(img_annotated, "Hoop Center", (h_center[0] + 10, h_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Uložení označeného snímku
    annotated_path = os.path.join(output_dir, "annotated_hoop_and_base.png")
    cv2.imwrite(annotated_path, img_annotated)
    print(f"Označený snímek byl uložen do: {annotated_path}")

else:
    print("Nepodařilo se získat snímek z kamery.")