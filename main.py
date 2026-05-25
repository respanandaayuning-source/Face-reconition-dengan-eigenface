import tkinter as tk
from gui import FaceRecognitionGUI
from face_recognizer import FaceRecognizer
import time

def main():
    root = tk.Tk()
    recognizer = FaceRecognizer(threshold=3500.0)#Untuk batas pengenalan

    def run_recognition():
        if not app.selected_dataset_path:
            return
        if not app.test_image_path:
            return

        #Untuk pengatur waktu
        start = time.time()
        recognizer.train(app.selected_dataset_path)
        label, dist, matched_path, status = recognizer.recognize(app.test_image_path)
        elapsed = time.time() - start

        result_label = label if label != "Unknown" else "Unknown (Tidak Dikenali)"
        app.update_display_metrics(result_label, dist, recognizer.threshold, elapsed, matched_path)

    app = FaceRecognitionGUI(root, run_callback=run_recognition)
    root.mainloop()

if __name__ == "__main__":
    main()