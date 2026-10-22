from ctu_mitsubishi import Rv6s
import cv2
import os
import numpy as np

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Získání snímku z kamery

q = robot.get_q()
q[0] += np.deg2rad(5)
robot.move_to_q(q)
robot.wait_for_motion_stop()

img = robot.grab_image()

q = robot.get_q()

position = robot.fk(q)

# Ověření, zda byl snímek úspěšně získán
# Ověření, zda byl snímek úspěšně získán
if img is not None:
    print(f"Úspěšně získáno foto o rozměrech: {img.shape}")
    
    # Vytvoření složky 'photos_hoops', pokud ještě neexistuje
    os.makedirs('photos_hoops', exist_ok=True)

    # Uložení obrázku
    success = cv2.imwrite('photos_hoops/hoop6.png', img)

    if success:
        print("Obrázek byl úspěšně uložen.")
    else:
        print("Chyba při ukládání obrázku pomocí cv2.imwrite.")

    # Uložení pozice robota do textového souboru
    with open('photos_hoops/hoop6.txt', 'w') as f:
        f.write(str(position))
    print("Pozice robota byla úspěšně uložena do textového souboru.")

else:
    print("Nepodařilo se získat snímek z kamery (img je None).")

# Uzavření připojení
robot.stop_robot()
robot.close_connection()