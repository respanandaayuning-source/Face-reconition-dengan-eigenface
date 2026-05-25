import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# Palet Warna 
BG_MAIN      = "#FFF0F7"   
BG_CARD      = "#FFFFFF"
BG_SIDEBAR   = "#FFE4F3"
ACCENT_PINK  = "#F472B6"
ACCENT_LILAC = "#C084FC"
ACCENT_MINT  = "#588272"
ACCENT_PEACH = "#FDBA74"
TEXT_DARK    = "#4B2C47"
TEXT_MID     = "#9D6B8A"
TEXT_LIGHT   = "#C9A5BD"
BTN_HOVER    = "#FCE7F3"

# Font
FONT_TITLE   = ("Kristen ITC", 20, "bold")
FONT_LABEL   = ("Kristen ITC", 10, "bold")
FONT_SMALL   = ("Segoe Script", 9)
FONT_RESULT  = ("Segoe Script", 15, "bold")
FONT_METRIC  = ("Segoe Script", 10)

# Helper image placeholder 
def make_placeholder(w=280, h=280):
    img = Image.new("RGB", (w, h), "#FFF0F7")
    draw = ImageDraw.Draw(img)
    import random
    random.seed(42)
    dots = [(c, r) for c in range(20, w, 40) for r in range(20, h, 40)]
    colors = ["#F9A8D4", "#DDD6FE", "#A7F3D0", "#FDE68A", "#FBCFE8"]
    for i, (cx, cy) in enumerate(dots):
        col = colors[i % len(colors)]
        draw.ellipse([cx-6, cy-6, cx+6, cy+6], fill=col, outline=None)
    cx, cy = w//2, h//2
    draw.rounded_rectangle([cx-35, cy-28, cx+35, cy+28], radius=8, outline="#F472B6", width=2)
    draw.ellipse([cx-14, cy-14, cx+14, cy+14], outline="#F472B6", width=2)
    draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill="#F472B6")
    draw.rectangle([cx+20, cy-28, cx+35, cy-18], fill="#FFF0F7")
    draw.ellipse([cx+22, cy-28, cx+32, cy-18], outline="#F472B6", width=2)
    return img

# Helper tombol pastel
class PastelButton(tk.Canvas):
    def __init__(self, parent, text, command, color=ACCENT_PINK,
                 text_color="white", width=200, height=38, radius=19, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent.cget("bg"), highlightthickness=0, **kwargs)
        self.command = command
        self.color = color
        self.hover_color = self._lighten(color)
        self.text_str = text
        self.w, self.h, self.r = width, height, radius

        self._draw(self.color)
        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.color))
        self.bind("<Button-1>", lambda e: self.command())

    def _lighten(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw(self, color):
        self.delete("all")
        r = self.r
        w, h = self.w, self.h
        self.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=color, outline="")
        self.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=color, outline="")
        self.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=color, outline="")
        self.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=color, outline="")
        self.create_rectangle(r, 0, w-r, h, fill=color, outline="")
        self.create_rectangle(0, r, w, h-r, fill=color, outline="")
        self.create_text(w//2, h//2, text=self.text_str,
                         fill="white", font=("Kristen ITC", 10, "bold"))


# Kelas GUI Utama
class FaceRecognitionGUI:
    def __init__(self, root, run_callback):
        self.root = root
        self.root.title("✿ Face Recognition ✿")
        self.root.state("zoomed")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(True, True)

        self.run_callback = run_callback
        self.selected_dataset_path = ""
        self.test_image_path = None
        self.img_test_display = None
        self.img_result_display = None
        self._ph_placeholder = None

        self.create_widgets()

    def create_widgets(self):
        self._build_header()
        self._build_body()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=ACCENT_PINK, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🌸", font=("Arial", 22), bg=ACCENT_PINK).pack(side="left", padx=18)
        tk.Label(hdr, text="✿  Face Recognition  ✿",
                 font=FONT_TITLE, bg=ACCENT_PINK, fg="white").pack(side="left", expand=True)
        tk.Label(hdr, text="🌸", font=("Arial", 22), bg=ACCENT_PINK).pack(side="right", padx=18)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=18, pady=14)
        self._build_sidebar(body)
        self._build_image_panels(body)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=BG_SIDEBAR, width=255, relief="flat", bd=0)
        sidebar.pack(side="left", fill="y", padx=(0, 14))
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="˚₊· ͟͟͞͞➳❥ Upload File",
                 font=("Kristen ITC", 11, "bold"), bg=BG_SIDEBAR,
                 fg=ACCENT_LILAC).pack(anchor="w", padx=14, pady=(14, 4))
        self._divider(sidebar)

        # Dataset
        self._section_label(sidebar, "① Dataset Folder")

        

        self.lbl_current_dataset = tk.Label(
            sidebar, text="Belum dipilih~", font=FONT_SMALL,
            bg=BG_SIDEBAR, fg=TEXT_MID, wraplength=220, justify="left")
        self.lbl_current_dataset.pack(anchor="w", padx=14, pady=(0, 6))

        btn_ds = PastelButton(sidebar, "📁  Pilih Folder Dataset",
                              self.browse_dataset_folder,
                              color=ACCENT_LILAC, width=225)
        btn_ds.pack(padx=14, pady=(0, 10))
        self._divider(sidebar)

        # Test image
        self._section_label(sidebar, "② Test Image")
        self.lbl_file_chosen = tk.Label(
            sidebar, text="Belum dipilih~", font=FONT_SMALL,
            bg=BG_SIDEBAR, fg=TEXT_MID, wraplength=220, justify="left")
        self.lbl_file_chosen.pack(anchor="w", padx=14, pady=(0, 6))

        btn_img = PastelButton(sidebar, "🖼  Pilih Gambar",
                               self.browse_test_image,
                               color=ACCENT_PEACH, width=225)
        btn_img.pack(padx=14, pady=(0, 10))
        self._divider(sidebar)

        btn_run = PastelButton(sidebar, "✨  Run Recognition!",
                               self.run_callback,
                               color=ACCENT_PINK, width=225, height=44)
        btn_run.pack(padx=14, pady=12)
        self._divider(sidebar)

        # Result
        self._section_label(sidebar, "Hasil")
        self.lbl_result_name = tk.Label(
            sidebar, text="( belum diproses~ )",
            font=FONT_RESULT, bg=BG_SIDEBAR, fg=ACCENT_PINK, wraplength=220)
        self.lbl_result_name.pack(anchor="w", padx=14, pady=(2, 8))

        mbox = tk.Frame(sidebar, bg="#FFD6EC", bd=0)
        mbox.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(mbox, text="Metrics",
                 font=("Kristen ITC", 9, "bold"), bg="#FFD6EC",
                 fg=TEXT_MID).pack(anchor="w", pady=(6, 2))

        self.lbl_metric_threshold = tk.Label(
            mbox, text="Threshold  : —",
            font=FONT_METRIC, bg="#FFD6EC", fg=TEXT_DARK)
        self.lbl_metric_threshold.pack(anchor="w", padx=8, pady=1)

        self.lbl_metric_distance = tk.Label(
            mbox, text="Distance   : —",
            font=FONT_METRIC, bg="#FFD6EC", fg=TEXT_DARK)
        self.lbl_metric_distance.pack(anchor="w", padx=8, pady=(1, 6))

    def _build_image_panels(self, parent):
        panels = tk.Frame(parent, bg=BG_MAIN)
        panels.pack(side="left", fill="both", expand=True)

        left_p = self._image_card(panels, "🌷  Test Image")
        left_p.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_p = self._image_card(panels, "🌼  Closest Result")
        right_p.pack(side="left", fill="both", expand=True)

        self.canvas_test = tk.Label(left_p, bg="#FFF0F7", relief="flat", bd=0)
        self.canvas_test.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.canvas_result = tk.Label(right_p, bg="#FFF0F7", relief="flat", bd=0)
        self.canvas_result.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.clear_displays()

    def _image_card(self, parent, title):
        card = tk.Frame(parent, bg=BG_CARD,
                        highlightbackground=ACCENT_PINK, highlightthickness=2)
        tk.Label(card, text=title, font=FONT_LABEL,
                 bg=BG_CARD, fg=ACCENT_PINK).pack(pady=(10, 4))
        return card

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=BG_SIDEBAR, height=36)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        self.lbl_exec_time = tk.Label(
            footer, text="⏱ Execution time : —",
            font=("Segoe Script", 9), bg=BG_SIDEBAR, fg=TEXT_MID)
        self.lbl_exec_time.pack(side="left", padx=20, pady=8)
        tk.Label(footer, text="made with  ♡",
                 font=("Segoe Script", 9), bg=BG_SIDEBAR, fg=TEXT_LIGHT).pack(side="right", padx=20)

    def _section_label(self, parent, text):
        tk.Label(parent, text=text, font=FONT_LABEL,
                 bg=BG_SIDEBAR, fg=TEXT_DARK).pack(anchor="w", padx=14, pady=(10, 2))

    def _divider(self, parent):
        tk.Frame(parent, bg=ACCENT_PINK, height=1).pack(fill="x", padx=14, pady=4)

    def clear_displays(self):
        ph = make_placeholder(280, 280)
        ph_img = ImageTk.PhotoImage(ph)
        self._ph_placeholder = ph_img
        self.canvas_test.config(image=ph_img)
        self.canvas_test.image = ph_img
        self.canvas_result.config(image=ph_img)
        self.canvas_result.image = ph_img

    # Event Handlers 

    def browse_dataset_folder(self):
        folder_path = filedialog.askdirectory(title="Pilih Folder Dataset")
        if not folder_path:
            return

        items = os.listdir(folder_path)

        # Hitung foto langsung di root
        root_imgs = [
            f for f in items
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # Hitung foto dari subfolder
        subfolders = [d for d in items if os.path.isdir(os.path.join(folder_path, d))]
        subfolder_imgs = []
        valid_persons = set()
        for sub in subfolders:
            imgs = [f for f in os.listdir(os.path.join(folder_path, sub))
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if imgs:
                subfolder_imgs.extend(imgs)
                valid_persons.add(sub)

        # Tambah label dari foto root
        for f in root_imgs:
            base = os.path.splitext(f)[0]
            valid_persons.add(base.split('_')[0] if '_' in base else base)

        total_photos = len(root_imgs) + len(subfolder_imgs)

        if total_photos == 0:
            messagebox.showwarning(
                "Folder Kosong",
                "Tidak ada gambar (.jpg/.png) ditemukan!\n\n"
                "Bisa pakai salah satu atau campuran:\n"
                "• Foto langsung di folder  → alex_1.jpg\n"
                "• Subfolder per orang      → alex/foto1.jpg"
            )
            return

        self.selected_dataset_path = folder_path
        name = os.path.basename(folder_path)
        self.lbl_current_dataset.config(
            text=f"✓ {name}\n"
                 f"{len(valid_persons)} orang • {total_photos} foto",
            fg=ACCENT_MINT
        )

    def browse_test_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.test_image_path = file_path
            fname = os.path.basename(file_path)
            self.lbl_file_chosen.config(text=f"✓ {fname}", fg=ACCENT_MINT)
            img = Image.open(file_path).resize((280, 280), Image.Resampling.LANCZOS)
            self.img_test_display = ImageTk.PhotoImage(img)
            self.canvas_test.config(image=self.img_test_display)
            self.canvas_test.image = self.img_test_display

    # Metrics 
    def update_display_metrics(self, result_label, distance, threshold,
                                exec_duration, matched_orig_path=None):
        self.lbl_exec_time.config(
            text=f"⏱ Execution time : {exec_duration:.4f} detik")
        self.lbl_metric_threshold.config(
            text=f"Threshold  : {threshold:.2f}")

        if result_label != "Unknown (Tidak Dikenali)":
            self.lbl_result_name.config(
                text=f"✅  {result_label}", fg=ACCENT_MINT)
            self.lbl_metric_distance.config(
                text=f"Distance   : {distance:.2f}", fg=ACCENT_MINT)
        else:
            self.lbl_result_name.config(
                text="❌  Unknown", fg="#F87171")
            self.lbl_metric_distance.config(
                text=f"Distance   : {distance:.2f}", fg="#F87171")

        if matched_orig_path and result_label != "Unknown (Tidak Dikenali)":
            img_res = Image.open(matched_orig_path).resize(
                (280, 280), Image.Resampling.LANCZOS)
            self.img_result_display = ImageTk.PhotoImage(img_res)
            self.canvas_result.config(image=self.img_result_display)
            self.canvas_result.image = self.img_result_display
        else:
            ph = make_placeholder(280, 280)
            ph_img = ImageTk.PhotoImage(ph)
            self.canvas_result.config(image=ph_img)
            self.canvas_result.image = ph_img