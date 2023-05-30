import cv2
import numpy as np
import time
import   matplotlib.pyplot as plt

start_time = time.time()

def stereo_matching(img_left, img_right, window_size=5, max_disparity=16):
    # Convertir les images en niveaux de gris
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Obtenir les dimensions de l'image de référence
    height, width = img_left_gray.shape

    # Créer une carte de disparité vide
    disparity_map = np.zeros((height, width), np.uint8)

    # Calculer la moitié de la taille de la fenêtre
    half_window = window_size // 2

    # Parcourir chaque pixel de l'image de référence
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            min_diff = float('inf')
            best_disparity = 0

            # Parcourir les différentes disparités possibles
            for d in range(max_disparity):
                diff = 0

                # Parcourir la fenêtre autour du pixel de référence
                for i in range(-half_window, half_window + 1):
                    for j in range(-half_window, half_window + 1):
                        # Calculer la différence de pixels entre les deux images
                        diff += abs(int(img_left_gray[y + i, x + j]) - int(img_right_gray[y + i, x + j - d]))

                # Mettre à jour la disparité minimale
                if diff < min_diff:
                    min_diff = diff
                    best_disparity = d

            # Assigner la meilleure disparité au pixel correspondant dans la carte de disparité
            disparity_map[y, x] = best_disparity

    # Appliquer un filtre médian pour lisser la carte de disparité
    disparity_map = cv2.medianBlur(disparity_map, 5)

    return disparity_map

def compute_quality(disparity_map, s):
    height, width = disparity_map.shape[:2]
    total_pixels = height * width
    error_pixels = np.sum(disparity_map > s)
    quality = (error_pixels / total_pixels) * 100
    return quality




# Charger les images gauche et droite
img_left = cv2.imread('cones/im2.png')
img_right = cv2.imread('cones/im6.png')

# Appeler la fonction de mise en correspondance stéréo
disparity_map = stereo_matching(img_left, img_right)

"""# Afficher la carte de disparité
cv2.imshow('Disparity Map', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Exemple d'utilisation
s_values = [1, 2]
for s in s_values:
    quality = compute_quality(disparity_map, s)
    print(f"Qualité pour s={s}: {quality}%")"""


quality_values = []
execution_times = []

# Exécutez votre algorithme pour différentes valeurs de bloc et plage de disparité
block_sizes = [5, 9, 13]
disparity_ranges = [8, 16, 32]

for block_size in block_sizes:
    for disparity_range in disparity_ranges:
        start_time = time.time()
        disparity_map = stereo_matching(img_left, img_right, block_size, disparity_range)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000

        quality = compute_quality(disparity_map, 1)  # Utilisez s=1 pour le graphe

        quality_values.append(quality)
        execution_times.append(execution_time)

# Tracer le graphe
plt.plot(execution_times, quality_values, 'o-')
plt.xlabel('Temps d\'exécution (ms)')
plt.ylabel('Qualité')
plt.title('Ratio Qualité/Rapidité')
plt.grid(True)
plt.show()



end_time = time.time()
execution_time = (end_time - start_time) * 1000  # Convertir en millisecondes
print(f"Temps d'exécution: {execution_time} ms")

