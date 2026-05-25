import cv2
import numpy as np

# Menggunakan model Haar Cascade bawaan OpenCV untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path, target_size=(100, 100)):
    # Untuk membaca gambar, mendeteksi wajah, memotong, dan meresize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        #Agar jika tidak terdeteksi, diatur ulang ke tengah
        resized = cv2.resize(img, target_size)
        return resized
    
    # Untuk mengambil wajah pertama yang terdeteksi
    (x, y, w, h) = faces[0]
    face_crop = img[y:y+h, x:x+w]
    
    #Untuk mengatur ulang ke ukuran standar agar vektor fitur panjangnya sama
    face_resized = cv2.resize(face_crop, target_size)
    return face_resized