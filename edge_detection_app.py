import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# App class
# =========================
class EdgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Preprocessing & Edge Detection (Tiền xử lý & Phát hiện biên)")

        self.image_path = None

        # ===== LEFT: Control panel =====
        control = tk.Frame(root, width=250)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        tk.Button(control, text="Open Image (Mở ảnh)", command=self.open_image).pack(fill="x")
        tk.Button(control, text="Save Result (Lưu ảnh)", command=self.save_image).pack(fill="x", pady=5)

        self.brightness = tk.Scale(
            control, from_=0, to=100, orient=tk.HORIZONTAL,
            label="Brightness (Độ sáng)", command=lambda x: self.update()
        )
        self.brightness.set(30)
        self.brightness.pack(fill="x")

        self.blur = tk.Scale(
            control, from_=1, to=15, orient=tk.HORIZONTAL,
            label="Gaussian Blur (Lọc & khử nhiễu)", command=lambda x: self.update()
        )
        self.blur.set(5)
        self.blur.pack(fill="x")

        self.lap_ksize = tk.Scale(
            control, from_=1, to=7, orient=tk.HORIZONTAL,
            label="Laplacian Kernel Size", command=lambda x: self.update()
        )
        self.lap_ksize.set(3)
        self.lap_ksize.pack(fill="x")

        self.sobel_ksize = tk.Scale(
            control, from_=1, to=7, orient=tk.HORIZONTAL,
            label="Sobel Kernel Size", command=lambda x: self.update()
        )
        self.sobel_ksize.set(3)
        self.sobel_ksize.pack(fill="x")

        # ===== RIGHT: Figure =====
        self.fig, self.ax = plt.subplots(2, 2, figsize=(8, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.img_original = None
        self.img_pre = None
        self.img_lap = None
        self.img_sobel = None

    # =========================
    def open_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
        )
        if self.image_path:
            self.update()

    # =========================
    def update(self):
        if self.image_path is None:
            return

        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b = self.brightness.get()
        k = self.blur.get()
        lk = self.lap_ksize.get()
        sk = self.sobel_ksize.get()

        if k % 2 == 0: k += 1
        if lk % 2 == 0: lk += 1
        if sk % 2 == 0: sk += 1

        bright = cv2.convertScaleAbs(gray, beta=b)
        blur = cv2.GaussianBlur(bright, (k, k), 0)

        lap = cv2.Laplacian(blur, cv2.CV_64F, ksize=lk)
        lap = np.uint8(np.abs(lap))

        sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sk)
        sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sk)
        sobel = np.uint8(np.sqrt(sx**2 + sy**2))

        self.img_original = img
        self.img_pre = blur
        self.img_lap = lap
        self.img_sobel = sobel

        titles = [
            "Original Image (Ảnh gốc)",
            "Preprocessed Image (Tiền xử lý)",
            "Laplacian Edge (Biên Laplacian)",
            "Sobel Edge (Biên Sobel)"
        ]
        images = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            blur, lap, sobel
        ]

        for i in range(2):
            for j in range(2):
                self.ax[i, j].clear()
                self.ax[i, j].imshow(images[i*2+j], cmap="gray")
                self.ax[i, j].set_title(titles[i*2+j])
                self.ax[i, j].axis("off")

        self.canvas.draw()

    # =========================
    def save_image(self):
        choice = simpledialog.askstring(
            "Choose Image (Chọn ảnh)",
            "original / pre / laplacian / sobel"
        )

        if not choice:
            return

        choice = choice.lower()
        img = {
            "original": self.img_original,
            "pre": self.img_pre,
            "laplacian": self.img_lap,
            "sobel": self.img_sobel
        }.get(choice)

        if img is None:
            messagebox.showerror("Error", "Invalid choice")
            return

        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            cv2.imwrite(path, img)
            messagebox.showinfo("Saved", "Image saved successfully")

# =========================
root = tk.Tk()
app = EdgeApp(root)
root.mainloop()
