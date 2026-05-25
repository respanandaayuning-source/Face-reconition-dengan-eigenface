import numpy as np

def hitung_eigen(L, k):
    num_samples = L.shape[0]
    A = L.copy().astype(float)
    eigenvalues = []
    eigenvectors = []
    
    for _ in range(k):
        v = np.ones(num_samples)
        v = v / np.linalg.norm(v) # Normalisasi vektor
        
        # Power iteration loop untuk mencari arah eigenvector
        for _ in range(50):
            # Mengganti looping for-for-for dengan perkalian matriks '@'
            Av = A @ v 
            norm_Av = np.linalg.norm(Av)
            
            if norm_Av < 1e-9:
                break
            v = Av / norm_Av
        
        # Hitung Eigenvalue (Rayleigh Quotient)
        lam = v.T @ A @ v
        
        eigenvalues.append(lam)
        eigenvectors.append(v)
        
        # Deflasi Matriks dengan menggunakan operasi outer product yang cepat
        A -= lam * np.outer(v, v)
                
    return np.array(eigenvalues), np.array(eigenvectors).T


def calculate_eigenfaces(X, num_components=None):
    num_samples, num_features = X.shape
    
    if num_components is None:
        num_components = num_samples - 1
        
    # Menghitung rata-rata wajah
    mean_face = np.mean(X, axis=0)
    
    # Mengurangi setiap wajah dengan rata-rata
    phi = X - mean_face
    
    # Menghitung matriks kovarians (L = phi @ phi.T) ukuran N x N
    L = np.dot(phi, phi.T)

    # Memetakan eigen value dan eigen vector
    eigenvalue, eigenvector_L = hitung_eigen(L, num_components)
        
    # Konversi eigen vector dari matriks L agar kembali ke ruang fitur asli 
    eigenvector = np.dot(phi.T, eigenvector_L)
    
    # Normalisasi panjang eigenvectors menjadi 1
    norm = np.linalg.norm(eigenvector, axis=0)
    norm[norm == 0] = 1 
    eigenvector = eigenvector / norm
    
    # Ambil top 'k' komponen
    top_eigenvector = eigenvector[:, :num_components]
    
    return mean_face, top_eigenvector