import cv2
import numpy as np
import os
from time import sleep, time
from ctu_mitsubishi import Rv6s


# Bezpečná výchozí (domovská) poloha robota v metrech[cite: 3, 4]
Home_position = np.array([
    [1,  0,  0,  0.4],
    [0, -1,  0, -0.3],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]
])


def get_ik_nearest(robot, position, q_ref):
    """
    Spočítá inverzní kinematiku (IK) pro cílovou matici 'position'[cite: 3, 4].
    Vybere konfiguraci kloubů nejbližší předchozí poloze 'q_ref'[cite: 3, 4].
    Vrací None, pokud řešení neexistuje[cite: 4].
    """
    if position is None:
        return None
    qs = robot.ik(position)
    if qs is None or len(qs) == 0:
        return None
    return min(qs, key=lambda q: np.linalg.norm(q - q_ref))


def check_trajectory_collision(robot, q_start, q_end, steps=10):
    """
    Zkontroluje přechod mezi kloubovými konfiguracemi q_start a q_end.
    Vrací (True, duvod_kolize) pokud hrozí kolize, jinak (False, "").
    """
    # 1. Kontrola skokového přetočení kloubů (ochrana proti singularitě a švihnutí ramene)
    # max_joint_diff = np.max(np.abs(q_end - q_start))
    # if max_joint_diff > 1.5:  # cca 86 stupňů v jednom kroku
    #     return True, f"Nebezpečný skok kloubu o {np.degrees(max_joint_diff):.1f}° (riziko kolize při přetočení)."

    # # 2. Kontrola mezibodů pomocí dopředné kinematiky (FK)
    for i in range(1, steps + 1):
        alpha = i / steps
        q_interp = (1.0 - alpha) * q_start + alpha * q_end
        T_fk = robot.fk(q_interp)
        
        p_gripper = T_fk[0:3, 3]
        R_gripper = T_fk[0:3, 0:3]
        p_circle = p_gripper + R_gripper @ np.array([0.135, 0.0, 0.030])

        # A) Kolize se základnou / stolem (base leží v Z = 7.0 cm)
        if p_circle[2] < 0.085 or p_gripper[2] < 0.115:
            return True, f"Nízká výška nad stolem (Z_obruč={p_circle[2]*100:.1f} cm, Z_gripper={p_gripper[2]*100:.1f} cm)."

        # B) Autokolize: zápěstí příliš blízko ke sloupu / základně robota v rovině XY
        dist_from_base_xy = np.hypot(p_gripper[0], p_gripper[1])
        if dist_from_base_xy < 0.22:
            return True, f"Gripper příliš blízko tělu robota (r={dist_from_base_xy*100:.1f} cm < 22 cm)."

    return False, ""


def move_bot(robot, position):
    """Okamžitý blokující pohyb na zadanou pozici (pro Home a parkování nad střed)[cite: 3, 4]."""
    q0 = robot.get_q()
    qnew = get_ik_nearest(robot, position, q0)
    if qnew is None:
        print(f"[CHYBA IK] Pro pozici X={position[0,3]:.3f}, Y={position[1,3]:.3f}, Z={position[2,3]:.3f} neexistuje řešení[cite: 4].")
        return False
    
    collision, reason = check_trajectory_collision(robot, q0, qnew)
    if collision:
        safe_exit(robot, f"Hrozí kolize při přípravném pohybu: {reason}")

    robot.move_to_q(qnew)
    robot.wait_for_motion_stop(poll_interval=0.01)
    return True

def move_home(robot):
    """Bezpečný návrat do Home (nejprve zvedne nástroj nad Z = 32 cm, aby nenarazil do tyče)[cite: 3, 4]."""
    print("Stěhování do domovské polohy (Home)...")
    q0 = robot.get_q()
    T_current = robot.fk(q0)
    if T_current[2, 3] < 0.32:
        T_lift = T_current.copy()
        T_lift[2, 3] = 0.32
        if not move_bot(robot, T_lift):
            return False
    return move_bot(robot, Home_position)






# ==========================================
if __name__ == "__main__":
    robot = Rv6s(debug=False)
    robot.initialize()

    # ---------------------------------------------------------
    # FÁZE 1 (t = 0 s): Přesun robota do Home pozice
    # ---------------------------------------------------------
    if not move_home(robot):
        safe_exit(robot, "Chyba při přesunu do Home position.")

    robot.stop_robot()
    robot.close_connection()