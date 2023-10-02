from PIL import Image
import os

# Directorio que contiene las imágenes TIFF
directorio_tiff = "validation/masks"

# Directorio de salida para las imágenes JPEG
directorio_jpg = "validation_masks"

# Asegúrate de que el directorio de salida exista
if not os.path.exists(directorio_jpg):
    os.makedirs(directorio_jpg)

# Asegúrate de que el directorio de salida exista
if not os.path.exists(directorio_jpg):
    os.makedirs(directorio_jpg)

counter = 0

# Recorre el directorio de TIFF y convierte las imágenes a JPEG
for archivo_tiff in os.listdir(directorio_tiff):
    if archivo_tiff.endswith(".tif") or archivo_tiff.endswith(".tiff"):
        counter += 1
        ruta_completa_tiff = os.path.join(directorio_tiff, archivo_tiff)
        nombre_sin_extension = os.path.splitext(archivo_tiff)[0]
        ruta_completa_jpg = os.path.join(directorio_jpg, "mask" + str(counter) + ".jpg")

        # Abre la imagen TIFF, conviértela al modo RGB y guárdala como JPEG
        imagen = Image.open(ruta_completa_tiff)
        imagen_rgb = imagen.convert("RGB")
        imagen_rgb.save(ruta_completa_jpg, "JPEG")

print("La conversión de TIFF a JPEG ha sido completada.")