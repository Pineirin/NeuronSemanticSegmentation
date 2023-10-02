import cv2

# Carga la imagen
imagen = cv2.imread('dataset/tm/train_masks/mask1.jpg')

# Verifica la cantidad de canales de color
num_canales = imagen.shape[2]

if num_canales == 1:
    print("La imagen es en escala de grises.")
elif num_canales == 3:
    print("La imagen es a color (RGB).")
else:
    print("La imagen tiene un número de canales diferente de 1 y 3, puede ser una imagen en color con un canal adicional (por ejemplo, RGBA).")