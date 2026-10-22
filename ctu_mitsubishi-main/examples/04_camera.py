from ctu_mitsubishi import Rv6s
import cv2
import os

# Inicializace rozhraní robota
robot = Rv6s(debug=False)
robot.initialize()

# Získání snímku z kamery
img = robot.grab_image()

# Ověření, zda byl snímek úspěšně získán
if img is not None:
    print(f"Úspěšně získáno foto o rozměrech: {img.shape}")
    
    # Vytvoření složky 'photos', pokud ještě neexistuje
    os.makedirs('photos', exist_ok=True)
    
    # Uložení obrázku uvnitř podmínky
    success = cv2.imwrite('photos/vystupni_obrazek7.png', img)
    if success:
        print("Obrázek byl úspěšně uložen.")
    else:
        print("Chyba při ukládání obrázku pomocí cv2.imwrite.")
else:
    print("Nepodařilo se získat snímek z kamery (img je None).")

# Uzavření připojení
robot.close_connection()