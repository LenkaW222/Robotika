# import cv2
# import numpy as np
# from ctu_mitsubishi import Rv6s

# # --- Matice H: pixel -> (X, Y) v bázi robota [m], rovina z = 0.07 ---
# # Buď načíst z kalibrace:
# H = np.load("H.npy")
# # ...nebo vložit ručně:
# # H = np.array([[ ..., ..., ...],
# #               [ ..., ..., ...],
# #               [ ..., ..., ...]], dtype=np.float64)

# H = np.array([[-1.53301591e-05,-1.73638498e-04,5.44990724e-01],[-1.63585342e-04,3.25080681e-05,2.51539033e-01],[-3.72363582e-05,-2.39915008e-05,1.00000000e+00]])

# Z_PLATFORM = 0.07          # výška roviny značek v bázi robota [m]
# MARKER_SIZE = 0.040        # strana značky [m] (jen pro kontrolu)

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


# def markers_in_base(img, H):
#     """
#     Najde ArUco značky a vrátí jejich polohu v bázi robota.
#     Výstup: {id: {"xyz": [X, Y, Z], "yaw": rad, "corners": 4x2, "side_mm": float}}
#     """
#     gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     corners, ids, _ = detector.detectMarkers(gray)

#     result = {}
#     if ids is None:
#         return result

#     for c, i in zip(corners, ids.ravel()):
#         c_px = c.reshape(4, 2).astype(np.float32)

#         # rohy značky: pixely -> báze robota [m]
#         c_base = cv2.perspectiveTransform(c_px[None], H)[0]

#         center = c_base.mean(axis=0)

#         # natočení značky: směr hrany roh0 -> roh1 (osa X značky) v bázi
#         d = c_base[1] - c_base[0]
#         yaw = np.arctan2(d[1], d[0])

#         # kontrola měřítka: průměrná délka hrany by měla být ~40 mm
#         sides = np.linalg.norm(np.roll(c_base, -1, axis=0) - c_base, axis=1)

#         result[int(i)] = {
#             "xyz": np.array([center[0], center[1], Z_PLATFORM]),
#             "yaw": float(yaw),
#             "corners": c_base,
#             "side_mm": float(sides.mean() * 1000),
#         }
#     return result


# if __name__ == "__main__":
#     # Snímek z robota (jako v 05_homo.py)
#     robot = Rv6s(debug=False)
#     robot.initialize()
#     img = robot.grab_image()
#     robot.close_connection()

#     # Alternativa bez robota:
#     # img = cv2.imread("snimek.png")

#     if img is None:
#         raise SystemExit("Nepodařilo se získat snímek z kamery (img je None).")

#     markers = markers_in_base(img, H)
#     if not markers:
#         raise SystemExit("Žádné ArUco značky nenalezeny.")

#     for mid, m in sorted(markers.items()):
#         x, y, z = m["xyz"]
#         print(f"ID {mid}: X={x*1000:.1f} mm, Y={y*1000:.1f} mm, Z={z*1000:.1f} mm, "
#               f"yaw={np.degrees(m['yaw']):.1f}°, strana={m['side_mm']:.1f} mm")



from ctu_mitsubishi import Rv6s
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Získání snímku z kamery[cite: 1]
img = robot.grab_image()

robot.close_connection()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_size_mm = 40
marker_offsets = {1: (5, 5), 2: (80, 80)}
marker_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * marker_size_mm

# Matice homografie z vašeho zadání
H = np.array([
    [-1.53301591e-05, -1.73638498e-04, 5.44990724e-01],
    [-1.63585342e-04, 3.25080681e-05, 2.51539033e-01],
    [-3.72363582e-05, -2.39915008e-05,  1.00000000e+00]
])

if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detected = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, rejected = detected.detectMarkers(gray)

    if ids is not None:
        img_markers = img.copy()
        cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)

        print("Detekovaná ID značek:", ids.flatten())

        # 1. Transformace rohů do souřadnic robota
        # cv2.perspectiveTransform vyžaduje tvar (N, 1, 2)
        robot_markers_corners = []
        robot_markers_centers = {}

        for i, marker_id in enumerate(ids.flatten()):
            # Rohy konkrétní značky (tvar 4x2)
            c = corners[i].reshape(-1, 1, 2).astype(np.float32)
            
            # Transformace rohů pomocí homografie H
            robot_corners = cv2.perspectiveTransform(c, H).reshape(-1, 2)
            robot_markers_corners.append(robot_corners)

            # 2. Výpočet středu značky (průměr jejích 4 rohů v souřadnicích robota)
            center_robot = np.mean(robot_corners, axis=0)
            robot_markers_centers[marker_id] = center_robot

            print(f"\nZnačka ID {marker_id}:")
            print(f"  Rohy v souřadnicích robota:\n{robot_corners}")
            print(f"  Střed v souřadnicích robota: X = {center_robot[0]:.3f}, Y = {center_robot[1]:.3f}")

        # Příklad vizualizace původních bodů v obraze[cite: 1]
        img_pts_16 = np.concatenate([c.reshape(4, 2) for c in corners])
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.plot(*img_pts_16.T, "o", color="red", markersize=4)
        plt.title("Detekované rohy ArUco značek")
        plt.show()

    else:
        print("Žádné ArUco značky nebyly detekovány.")
else:
    print("Nepodařilo se získat snímek z kamery (img je None).[cite: 1]")