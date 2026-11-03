import os
import cv2
import numpy as np
from ctu_mitsubishi import Rv6s


def compute_homography_and_robot_coordinates(T_gripper, img, Z_base_mm=70.0):
    """
    Detekuje ArUco značky, vypočítá homografii H, promítne středy do obrázku
    a spočítá 3D souřadnice desky a značek v souřadnicovém systému robota.
    """
    # 1. Detekce ArUco značek v obrázku
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        raise ValueError("V obrázku nebyly detekovány žádné ArUco značky.")

    # 2. Lokální 2D souřadnice (v mm) rohů ArUco značky 40x40 mm na desce
    marker_square = np.array([
        [0.0, 0.0],
        [40.0, 0.0],
        [40.0, 40.0],
        [0.0, 40.0]
    ], dtype=np.float32)

    # Offsety levého horního rohu ArUco značek na desce 125x125 mm
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

    # 3. Výpočet 2D Homografie H
    H, _ = cv2.findHomography(world_pts, image_pts)

    # 4. 3D poloha středu kruhu v robotových souřadnicích (v mm)
    hoop_offset_gripper = np.array([0.0, 0.030, 0.135, 1.0])
    hoop_center_3d_m = T_gripper @ hoop_offset_gripper
    hoop_robot_xy_mm = hoop_center_3d_m[:2] * 1000.0

    # 5. Výpočet souřadnic v robotickém systému (v mm)
    board_center_local_mm = np.array([62.5, 62.5])
   
    base_center_robot_mm = np.array([
        hoop_robot_xy_mm[0] + (62.5 - hoop_robot_xy_mm[0]),
        hoop_robot_xy_mm[1] + (62.5 - hoop_robot_xy_mm[1]),
        Z_base_mm
    ])

    # Středy ArUco značek v lokálním systému desky (v mm)
    marker_centers_local_mm = {
        1: np.array([37.5, 37.5]),
        2: np.array([100.0, 100.0])
    }

    marker_robot_coords_mm = {}
    for m_id, m_center_local in marker_centers_local_mm.items():
        m_x_robot = hoop_robot_xy_mm[0] + (m_center_local[0] - hoop_robot_xy_mm[0])
        m_y_robot = hoop_robot_xy_mm[1] + (m_center_local[1] - hoop_robot_xy_mm[1])
        marker_robot_coords_mm[m_id] = np.array([m_x_robot, m_y_robot, Z_base_mm])

    # 6. Projekce středu desky a středu kruhu do obrázku
    board_center_mm = np.array([62.5, 62.5, 1.0])
    board_px_homog = H @ board_center_mm
    board_pixel = board_px_homog[:2] / board_px_homog[2]

    hoop_2d_board = np.array([hoop_robot_xy_mm[0], hoop_robot_xy_mm[1], 1.0])
    hoop_px_homog = H @ hoop_2d_board
    hoop_pixel = hoop_px_homog[:2] / hoop_px_homog[2]

    print("\n--- SOUŘADNICE V SOUŘADNICOVÉM SYSTÉMU ROBOTA ---")
    print(f"Střed základní desky (Base Center):")
    print(f"  [X, Y, Z] = [{base_center_robot_mm[0] / 1000.0:.4f}, {base_center_robot_mm[1] / 1000.0:.4f}, {base_center_robot_mm[2] / 1000.0:.4f}] m")
    print(f"  [X, Y, Z] = [{base_center_robot_mm[0]:.1f}, {base_center_robot_mm[1]:.1f}, {base_center_robot_mm[2]:.1f}] mm")
   
    for m_id, coords in marker_robot_coords_mm.items():
        print(f"Střed ArUco značky ID {m_id}:")
        print(f"  [X, Y, Z] = [{coords[0] / 1000.0:.4f}, {coords[1] / 1000.0:.4f}, {coords[2] / 1000.0:.4f}] m")
        print(f"  [X, Y, Z] = [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}] mm")

    return H, board_pixel, hoop_pixel


if __name__ == "__main__":
    # 1. Inicializace robota
    robot = Rv6s(debug=False)
    robot.initialize()


    # Získání aktuální polohy robota bez jakéhokoliv pohybu
    q = robot.get_q()
    T_current = robot.fk(q)

    # KROK 1: Pořízení snímku ze současné pozice robota
    img = robot.grab_image()

    if img is None:
        print("Nepodařilo se získat snímek z kamery.")
    else:
        print(f"Úspěšně pořízen snímek o rozměrech: {img.shape}")

        # KROK 2: Vytvoření složky photos_homo a uložení originálního snímku
        output_dir = "photos_homo"
        os.makedirs(output_dir, exist_ok=True)

        orig_path = os.path.join(output_dir, "original_image.png")
        cv2.imwrite(orig_path, img)
        print(f"Originální snímek uložen do: {orig_path}")

        # KROK 3 a 4: Výpočet homografie, souřadnic v robotu a promítnutí středů
        H, board_px, hoop_px = compute_homography_and_robot_coordinates(
            T_current, img, Z_base_mm=70.0
        )

        # Vykreslení středů do snímku
        img_annotated = img.copy()

        b_center = (int(round(board_px[0])), int(round(board_px[1])))
        h_center = (int(round(hoop_px[0])), int(round(hoop_px[1])))

        # Zelený křížek = Střed základní desky
        cv2.drawMarker(img_annotated, b_center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img_annotated, "Base Center", (b_center[0] + 10, b_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Červený křížek = Střed kruhu
        cv2.drawMarker(img_annotated, h_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img_annotated, "Hoop Center", (h_center[0] + 10, h_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Uložení označeného snímku do složky photos_homo
        annotated_path = os.path.join(output_dir, "annotated_hoop_and_base.png")
        cv2.imwrite(annotated_path, img_annotated)
        
        print(f"Anotovaný snímek uložen do: {annotated_path}")

    robot.stop_robot()
    robot.close_connection()