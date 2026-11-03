import cv2
import numpy as np
from ctu_mitsubishi import Rv6s

# --- Matice H: pixel -> (X, Y) v bázi robota [m], rovina z = 0.07 ---
# Buď načíst z kalibrace:
H = np.load("H.npy")
# ...nebo vložit ručně:
# H = np.array([[ ..., ..., ...],
#               [ ..., ..., ...],
#               [ ..., ..., ...]], dtype=np.float64)

H = np.array([[-1.53301591e-05,-1.73638498e-04,5.44990724e-01],[-1.63585342e-04,3.25080681e-05,2.51539033e-01],[-3.72363582e-05,-2.39915008e-05,1.00000000e+00]])

Z_PLATFORM = 0.07          # výška roviny značek v bázi robota [m]
MARKER_SIZE = 0.040        # strana značky [m] (jen pro kontrolu)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())


def markers_in_base(img, H):
    """
    Najde ArUco značky a vrátí jejich polohu v bázi robota.
    Výstup: {id: {"xyz": [X, Y, Z], "yaw": rad, "corners": 4x2, "side_mm": float}}
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    result = {}
    if ids is None:
        return result

    for c, i in zip(corners, ids.ravel()):
        c_px = c.reshape(4, 2).astype(np.float32)

        # rohy značky: pixely -> báze robota [m]
        c_base = cv2.perspectiveTransform(c_px[None], H)[0]

        center = c_base.mean(axis=0)

        # natočení značky: směr hrany roh0 -> roh1 (osa X značky) v bázi
        d = c_base[1] - c_base[0]
        yaw = np.arctan2(d[1], d[0])

        # kontrola měřítka: průměrná délka hrany by měla být ~40 mm
        sides = np.linalg.norm(np.roll(c_base, -1, axis=0) - c_base, axis=1)

        result[int(i)] = {
            "xyz": np.array([center[0], center[1], Z_PLATFORM]),
            "yaw": float(yaw),
            "corners": c_base,
            "side_mm": float(sides.mean() * 1000),
        }
    return result


if __name__ == "__main__":
    # Snímek z robota (jako v 05_homo.py)
    robot = Rv6s(debug=False)
    robot.initialize()
    img = robot.grab_image()
    robot.close_connection()

    # Alternativa bez robota:
    # img = cv2.imread("snimek.png")

    if img is None:
        raise SystemExit("Nepodařilo se získat snímek z kamery (img je None).")

    markers = markers_in_base(img, H)
    if not markers:
        raise SystemExit("Žádné ArUco značky nenalezeny.")

    for mid, m in sorted(markers.items()):
        x, y, z = m["xyz"]
        print(f"ID {mid}: X={x*1000:.1f} mm, Y={y*1000:.1f} mm, Z={z*1000:.1f} mm, "
              f"yaw={np.degrees(m['yaw']):.1f}°, strana={m['side_mm']:.1f} mm")