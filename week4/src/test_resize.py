import cv2 as cv
import os
import matplotlib.pyplot as plt
from keypoint_detection import sift, match

# Configuración de rutas
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.join(base_path, "./data/qsd1_w4/images_without_noise/masked")
folder_path_bbdd = os.path.join(base_path, "./data/BBDD")

# Ruta de la imagen de consulta
image_path_1 = os.path.join(folder_path, "00001_0.jpg")

# Leer y procesar la imagen de consulta
img1 = cv.imread(image_path_1)
kp1, des1 = sift(img1)

# Asegurarse de que hay descriptores en la imagen de consulta
if des1 is None:
    print("No se encontraron descriptores en la imagen de consulta.")
else:
    # Procesar cada imagen en la carpeta BBDD
    for img_filename in os.listdir(folder_path_bbdd):
        # Ruta de cada imagen en BBDD
        image_path_2 = os.path.join(folder_path_bbdd, img_filename)
        img2 = cv.imread(image_path_2)

        if img2 is None:
            continue  # Omitir archivos no válidos

        # Detección de puntos clave y descriptores en la imagen de BBDD
        kp2, des2 = sift(img2)

        # Verificar que se encontraron descriptores en la imagen de BBDD
        if des2 is None:
            print(f"No se encontraron descriptores en la imagen {img_filename}.")
            continue

        # Comparar descriptores sin redimensionar
        matches = match(des1, des2, 'sift')

        # Dibujar coincidencias sin redimensionar
        img12_original = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Redimensionar ambas imágenes
        img1_resized = cv.resize(img1, (256, 256))
        img2_resized = cv.resize(img2, (256, 256))
        kp1_resized, des1_resized = sift(img1_resized)
        kp2_resized, des2_resized = sift(img2_resized)

        # Verificar descriptores redimensionados
        if des1_resized is None or des2_resized is None:
            print(f"No se encontraron descriptores en la versión redimensionada de {img_filename}.")
            continue
        
        # Comparar descriptores redimensionados
        matches_resized = match(des1_resized, des2_resized, 'sift')

        # Dibujar coincidencias redimensionadas
        img12_resized = cv.drawMatches(img1_resized, kp1_resized, img2_resized, kp2_resized, matches_resized, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Convertir imágenes a RGB para visualización
        img12_original_rgb = cv.cvtColor(img12_original, cv.COLOR_BGR2RGB)
        img12_resized_rgb = cv.cvtColor(img12_resized, cv.COLOR_BGR2RGB)

        # Mostrar ambas versiones en una misma figura
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.imshow(img12_original_rgb)
        plt.title(f"Matches (Original) - {img_filename} - {len(matches)} matches")
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.imshow(img12_resized_rgb)
        plt.title(f"Matches (Resized) - {img_filename} - {len(matches_resized)} matches")
        plt.axis('off')
        
        plt.show()

