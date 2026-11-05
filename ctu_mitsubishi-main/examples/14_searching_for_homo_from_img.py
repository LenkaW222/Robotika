import os
import numpy as np
import cv2

# Povolené indexy referenčních obrázků
#ALLOWED_INDICES = {"01", "43", "42", "63", "71"}
ALLOWED_INDICES = {"11", "31", "32", "63", "73"}  # Tyhle img jsou lepší

# ==========================================
# 1. ČÁST: Detekce ArUco a výpočet středu base
# ==========================================

def get_aruco_centers(img):
    """
    Detekuje ArUco značky (ID1 a ID2) a spočítá jejich středy v pixelech.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners, ids, _ = detector.detectMarkers(gray)
    
    centers = {}
    if ids is not None:
        for idx, marker_id in enumerate(ids.flatten()):
            c = corners[idx].reshape(4, 2)
            center = np.mean(c, axis=0)
            centers[marker_id] = center
            
    return centers

def get_base_center_from_aruco(centers_dict):
    """
    Vypočítá střed base z pozic ArUco značek ID1 a ID2.
    """
    if 1 in centers_dict and 2 in centers_dict:
        return (centers_dict[1] + centers_dict[2]) / 2.0
    return None

# ==========================================
# 2. ČÁST: Detekce kruhu a výpočet homografie
# ==========================================

def parse_transformation_matrix(file_path):
    """
    Načte matici T (4x4) z textového souboru.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    matrix_data = []
    for line in lines:
        cleaned = line.replace('[', '').replace(']', '').strip()
        if cleaned:
            row = [float(num) for num in cleaned.split() if num]
            if len(row) == 4:
                matrix_data.append(row)
                
    return np.array(matrix_data, dtype=np.float32)

def compute_circle_world_pos(T_gripper):
    """
    Spočítá souřadnice středu kruhu v globálním souřadném systému robota [x, y, z] v mm.
    Offset vůči gripperu: +135 mm v X a -30 mm v Z.
    """
    p_circle_gripper = np.array([0.135, 0.0, -0.030, 1.0])
    p_circle_robot = T_gripper @ p_circle_gripper
    
    x_mm = p_circle_robot[0] * 1000.0
    y_mm = p_circle_robot[1] * 1000.0
    z_mm = p_circle_robot[2] * 1000.0
    
    return np.array([x_mm, y_mm, z_mm])

def detect_circle_image_pos(img):
    """
    Detekuje střed terče v pixelech pomocí přesně odladěných parametrů HoughCircles.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray_blur = cv2.medianBlur(gray, 5)
    
    # Vaše odladěné parametry
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=20,
        param1=50,
        param2=75,
        minRadius=100,
        maxRadius=120
    )
    
    if circles is not None:
        detected_circles = circles[0]
        best_circle = detected_circles[0]  # První/nejlepší nalezený kruh
        
        center_px = np.array([best_circle[0], best_circle[1]], dtype=np.float32)
        radius = float(best_circle[2])
        return center_px, radius
    
    return None, None

def build_homography_from_dataset(dataset_dir, allowed_indices=ALLOWED_INDICES):
    """
    Vypočítá matici homografie H z vybraných snímků.
    """
    if not os.path.exists(dataset_dir):
        return None, [], []

    file_list = os.listdir(dataset_dir)
    img_files = []
    
    for f in file_list:
        if f.startswith("img") and (f.endswith(".png") or f.endswith(".jpg")):
            name_no_ext = f.rsplit('.', 1)[0]
            num_part = name_no_ext.replace("img", "")
            if num_part in allowed_indices:
                img_files.append(f)
                
    img_files = sorted(img_files)
    
    pts_robot = []
    pts_image = []
    
    for img_name in img_files:
        img_path = os.path.join(dataset_dir, img_name)
        
        name_no_ext = img_name.rsplit('.', 1)[0]
        prefix_number = name_no_ext.replace("img", "")
        txt_name = f"positions{prefix_number}.txt"
        txt_path = os.path.join(dataset_dir, txt_name)
        
        if not os.path.exists(txt_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        T_gripper = parse_transformation_matrix(txt_path)
        pos_world = compute_circle_world_pos(T_gripper)
        pos_img, _ = detect_circle_image_pos(img)
        
        if pos_img is not None:
            pts_robot.append([pos_world[0], pos_world[1]])
            pts_image.append(pos_img)
            
    if len(pts_robot) < 4:
        print("Nedostatek korespondujících bodů pro výpočet homografie.")
        return None, np.array(pts_image), np.array(pts_robot)
        
    pts_robot = np.array(pts_robot, dtype=np.float32)
    pts_image = np.array(pts_image, dtype=np.float32)
    
    H, _ = cv2.findHomography(pts_image, pts_robot, cv2.RANSAC, 5.0)
    return H, pts_image, pts_robot

# ==========================================
# 3. ČÁST: Převod z Pixelů do Robot X,Y
# ==========================================

def transform_pixel_to_robot(point_pixel, H):
    """
    Převede bod [u, v] v pixelech na [X, Y] robota v mm pomocí matice H.
    """
    px_homo = np.array([point_pixel[0], point_pixel[1], 1.0], dtype=np.float32)
    robot_homo = H @ px_homo
    return np.array([robot_homo[0] / robot_homo[2], robot_homo[1] / robot_homo[2]])

# ==========================================
# Vykreslení a uložení anotovaných obrázků
# ==========================================

def annotate_and_save_images(dataset_dir, output_dir, H=None, allowed_indices=ALLOWED_INDICES):
    """
    Vytvoří výstupní složku a uloží do ní vybrané anotované obrázky.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_list = os.listdir(dataset_dir)
    img_files = []
    
    for f in file_list:
        if f.startswith("img") and (f.endswith(".png") or f.endswith(".jpg")):
            name_no_ext = f.rsplit('.', 1)[0]
            num_part = name_no_ext.replace("img", "")
            if num_part in allowed_indices:
                img_files.append(f)
                
    img_files = sorted(img_files)
    
    for img_name in img_files:
        img_path = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        annotated = img.copy()
        
        # 1. Střed kruhu (zeleně)
        circle_center, radius = detect_circle_image_pos(img)
        if circle_center is not None:
            cx, cy = int(circle_center[0]), int(circle_center[1])
            cv2.circle(annotated, (cx, cy), int(radius), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated, "Stred kruhu", (cx - 40, cy - int(radius) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 2. Střed base (modře)
        aruco_centers = get_aruco_centers(img)
        base_center_px = get_base_center_from_aruco(aruco_centers)
        
        if base_center_px is not None:
            bx, by = int(base_center_px[0]), int(base_center_px[1])
            cv2.circle(annotated, (bx, by), 7, (255, 0, 0), -1)
            cv2.putText(annotated, "Stred base", (bx - 40, by - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if H is not None:
                robot_pos = transform_pixel_to_robot(base_center_px, H)
                cv2.putText(annotated, f"Robot X,Y: [{robot_pos[0]:.1f}, {robot_pos[1]:.1f}] mm",
                            (bx - 80, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                            
        out_path = os.path.join(output_dir, f"annotated_{img_name}")
        cv2.imwrite(out_path, annotated)

# ==========================================
# MAIN EXECUTABLE
# ==========================================

if __name__ == "__main__":
    dataset_dir = "photos_3_11_26"
    output_dir = "annotated_results"
    
    if os.path.exists(dataset_dir):
        # 1. Výpočet matice homografie H
        H, img_pts, rob_pts = build_homography_from_dataset(dataset_dir)
        
        print("\n" + "="*50)
        print(" VÝSLEDKY HOMOGRAFIE A POLOHY BASE")
        print("="*50)
        
        if H is not None:
            print("\n1) Matice homografie H (převod: Pixel [u,v] -> Robot [X,Y] v mm):")
            print(np.array2string(H, formatter={'float_kind': lambda x: f"{x:12.6f}"}))
            
            # 2. Určení středu base na prvním dostupném snímku
            file_list = os.listdir(dataset_dir)
            sample_img_name = next((f for f in sorted(file_list) if f.startswith("img") and (f.endswith(".png") or f.endswith(".jpg"))), None)
            
            if sample_img_name:
                sample_img = cv2.imread(os.path.join(dataset_dir, sample_img_name))
                aruco_centers = get_aruco_centers(sample_img)
                base_center_px = get_base_center_from_aruco(aruco_centers)
                
                if base_center_px is not None:
                    base_center_robot = transform_pixel_to_robot(base_center_px, H)
                    print(f"\n2) Poloha středu base (z obrázku {sample_img_name}):")
                    print(f"   - Pixelové souřadnice (u, v): [{base_center_px[0]:.2f}, {base_center_px[1]:.2f}] px")
                    print(f"   - Souřadnice robota  (X, Y): [{base_center_robot[0]:.2f}, {base_center_robot[1]:.2f}] mm")
            
            # Uložení výstupních snímků
            annotate_and_save_images(dataset_dir, output_dir, H)
            print(f"\nAnotované obrázky byly uloženy do složky: '{output_dir}/'")
        else:
            print("Při výpočtu homografie došlo k chybě.")
        print("="*50 + "\n")
    else:
        print(f"Složka '{dataset_dir}' nebyla nalezena.")