import os
import numpy as np
from helper import detect_and_crop_face

def load_dataset(dataset_path):
    images = []
    labels = []
    original_paths = []
    
    # Agar program mampu menelusuri seluruh folder dan subfolder
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                
                #Agar nama folder atau file dapat digunakan sebagai label pada program
                label = os.path.basename(root) if root != dataset_path else os.path.splitext(file)[0]
                
                face_img = detect_and_crop_face(path)
                if face_img is not None:
                    #Mengubah gambar 2D menjadi vektor 1D
                    flattened = face_img.flatten()
                    images.append(flattened)
                    labels.append(label)
                    original_paths.append(path)
                    
    # Untuk mengonversi ke numpy array
    # X merupakan matriks dengan baris sebagai sampel gambar
    X = np.array(images, dtype=np.float32)
    return X, labels, original_paths