

import cv2
import os
import glob
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


#Task 1

# Ruta a la carpeta que contiene las imágenes
folder_path = "../data/qsd1_w3/qsd1_w3"

# Obtener todas las imágenes con extensión .jpg
image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))


# Aplicar filtro de mediana a cada imagen
for image_path in image_paths:
    # Leer la imagen
    img = cv2.imread(image_path)  
    
    # Aplicar el filtro de mediana con un tamaño de kernel de 5
    #median_filtered = cv2.medianBlur(img, 3)
    median_filtered = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 3,15)
    mse_value = mean_squared_error(img, median_filtered)
    psnr_value = peak_signal_noise_ratio(img,median_filtered)
    grayA = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(grayA, grayB, full=True)

   
    print(f"Image: {os.path.basename(image_path)},MSE: {mse_value:.2f}, PSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
    
    # Mostrar la imagen original y la imagen filtrada
    #cv2.imshow("Original", img)
    #cv2.imshow("Filtrada con Mediana", median_filtered)
    
    # Esperar a que se presione una tecla para continuar
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Cerrar todas las ventanas de OpenCV al final
#cv2.destroyAllWindows()