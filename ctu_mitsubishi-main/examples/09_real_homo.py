import os
import cv2
import numpy as np
from ctu_mitsubishi import Rv6s

def detect_hoop_center(img):
    """Detects the circular hoop in the image and returns its pixel coordinates (u, v) and radius."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # 1. Primary approach: Hough Circles detection
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=20, 
        maxRadius=80
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        hoop = circles[0][0]
        return float(hoop[0]), float(hoop[1]), float(hoop[2])
    
    # 2. Fallback: Contour thresholding & minimum enclosing circle
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 20 < radius < 80:
            return float(x), float(y), float(radius)

    return None, None, None


def compute_homography_and_hoop_position(image_path, text_path):
    """
    1. Loads the image and 4x4 matrix T (derived from joint angles q via forward kinematics).
    2. Detects ArUco markers to compute planar homography (pixels <-> board plane at Z=70mm).
    3. Finds the hoop center (u, v) in pixel space and transforms it to board coordinates.
    4. Computes 3D hoop center in the robot base frame using matrix T and the 90-degree local offset.
    """
    # Load image and Forward Kinematics Transformation Matrix T
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
        
    T_fk = np.loadtxt(text_path)  # 4x4 matrix computed from joint angles q

    # Detect ArUco markers (DICT_4X4_50)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        print(f"[{image_path}] No ArUco markers detected.")
        return

    # ArUco board setup (coordinates in mm on the Z = 70 mm plane)
    marker_size_mm = 40.0
    marker_offsets = {1: (5.0, 5.0), 2: (80.0, 80.0)}
    marker_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32) * marker_size_mm

    board_pts, img_pts = [], []
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in marker_offsets:
            offset = np.array(marker_offsets[marker_id], dtype=np.float32)
            board_pts.append(marker_square + offset)
            img_pts.append(corners[i].reshape(4, 2))

    board_pts = np.vstack(board_pts).astype(np.float32)
    img_pts = np.vstack(img_pts).astype(np.float32)

    # Compute Homography: Board (mm) <-> Image (pixels)
    H_board2img, _ = cv2.findHomography(board_pts, img_pts)
    H_img2board = np.linalg.inv(H_board2img)

    # Detect Hoop Center in Image
    u, v, r_px = detect_hoop_center(img)
    if u is None:
        print(f"[{image_path}] Hoop detection failed.")
        return

    print(f"\n--- Processing: {image_path} ---")
    print(f"Detected Hoop Center (Image Pixels): u = {u:.1f} px, v = {v:.1f} px (r = {r_px:.1f} px)")

    # Convert Image Pixels -> Board Plane (mm)
    pixel_vec = np.array([u, v, 1.0], dtype=np.float32)
    board_vec = H_img2board @ pixel_vec
    x_board_mm = board_vec[0] / board_vec[2]
    y_board_mm = board_vec[1] / board_vec[2]
    print(f"Hoop Center on Board Plane (at Z = 70 mm): X = {x_board_mm:.2f} mm, Y = {y_board_mm:.2f} mm")

    # 3D Kinematics Calculation
    # Normalize T matrix units to mm if required
    T_mm = T_fk.copy()
    if np.max(np.abs(T_fk[:3, 3])) < 10.0:
        T_mm[:3, 3] *= 1000.0

    # Local offset: 135 mm along Y-axis (90 degrees to X-pointing gripper)
    hoop_local_offset = np.array([0.0, 135.0, 0.0, 1.0])
    hoop_center_robot_3d = T_mm @ hoop_local_offset

    print(f"3D Hoop Center in Robot Base Frame:")
    print(f"  X_base = {hoop_center_robot_3d[0]:.2f} mm")
    print(f"  Y_base = {hoop_center_robot_3d[1]:.2f} mm")
    print(f"  Z_base = {hoop_center_robot_3d[2]:.2f} mm")


if __name__ == "__main__":
    # Execute for saved photo & pose files
    compute_homography_and_hoop_position('photos_hoop/hoop1.png', 'photos_hoop/hoop1.txt')
    compute_homography_and_hoop_position('photos_hoop/hoop2.png', 'photos_hoop/hoop2.txt')