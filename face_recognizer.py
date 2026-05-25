import numpy as np
from eigen_math import calculate_eigenfaces
from data_loader import load_dataset
from helper import detect_and_crop_face

class FaceRecognizer:
    # Untuk threshold bawaan/default
    def __init__(self, threshold=3500.0):
        self.threshold = threshold
        self.mean_face = None
        self.eigenvectors = None
        self.projected_weights = None
        self.labels = []
        self.original_paths = []
        
    def train(self, dataset_path):
        X, self.labels, self.original_paths = load_dataset(dataset_path)
        if len(X) == 0:
            return False
            
        self.mean_face, self.eigenvectors = calculate_eigenfaces(X)
        
        # Untuk royeksikan dataset ke ruang eigen dengan bobot/weights
        phi = X - self.mean_face
        self.projected_weights = np.dot(phi, self.eigenvectors)
        return True
        
    def recognize(self, image_path):
        if self.mean_face is None:
            return None, 0, None, "Model belum dilatih"
            
        face_img = detect_and_crop_face(image_path)
        if face_img is None:
            return None, 0, None, "Wajah tidak terdeteksi"
            
        # Untuk vektorisasi wajah yang dites dan mengurangi mean
        test_flattened = face_img.flatten()
        phi_test = test_flattened - self.mean_face
        
        # Untuk memproyeksikan wajah yang dites
        test_weight = np.dot(phi_test, self.eigenvectors)
        
        # Untuk menghitung euclidean distance ke semua bobot pada dataset
        distances = np.linalg.norm(self.projected_weights - test_weight, axis=1)
        
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        if min_dist < self.threshold:
            return self.labels[min_idx], min_dist, self.original_paths[min_idx], "Dikenali"
        else:
            return "Unknown", min_dist, None, "Melewati Threshold"