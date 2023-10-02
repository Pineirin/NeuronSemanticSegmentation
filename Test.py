import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

model = load_model('modelo_jpg.h5')

def predict_and_segment(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128)) 
    img = img / 255.0

    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    print("Min prediction value:", np.min(prediction))
    print("Max prediction value:", np.max(prediction))
    print("Mean prediction value:", np.mean(prediction))

    threshold = 0.5 
    mask = (prediction > threshold).astype(np.uint8)

    print("Unique values in mask:", np.unique(mask))
    print("Mask shape:", mask.shape)

    return mask

def main():
    input_image_path = 'dataset1_jpg/images/....jpg'

    segmentation_mask = predict_and_segment(input_image_path)

    cv2.imwrite('segmentation_mask.jpg', segmentation_mask * 255)


if __name__ == "__main__":
    # We want to use the relative path of the file (instead of the project one).
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    main()


