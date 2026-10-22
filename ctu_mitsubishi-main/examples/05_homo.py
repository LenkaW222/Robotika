from ctu_mitsubishi import Rv6s
import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Získání snímku z kamery
img = robot.grab_image()

robot.close_connection()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker_size_mm = 40
marker_offsets = {1: (5, 5), 2:(80, 80)}
marker_square = np.array([[0 ,0], [1, 0], [1, 1], [0,1]])*marker_size_mm



if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    detected = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, rejected = detected.detectMarkers(gray)

    img_markers = img.copy()
    cv2.aruco.drawDetectedMarkers(img_markers, corners, ids)

    print(ids)
    print(corners[0])


    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_markers)
    # plt.show()

    paper_pts_16 = np.concatenate([marker_square + marker_offsets[i[0]] for i in ids]).astype(np.float32)
    img_pts_16 = np.concatenate([c.reshape(4, 2) for c in corners])

    H_markers, _ = cv2.findHomography(paper_pts_16, img_pts_16)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.plot(*img_pts_16.T, "o", color="red", markersize=4)
    plt.show()

else:
    print("Nepodařilo se získat snímek z kamery (img je None).")

# Uzavření připojení