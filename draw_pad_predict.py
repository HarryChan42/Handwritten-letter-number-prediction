import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw, ImageOps
MODEL_PATH = "mnist_cnn.keras"
CANVAS_SIZE = 280
BRUSH_SIZE = 18
MNIST_SIZE = 28

#Model Trained from classification
model = tf.keras.models.load_model(MODEL_PATH)

class DigitPadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Pad (CNN Prediction)")

        # Main frame
        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(main, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black background
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        ttk.Button(main, text="Predict", command=self.predict).grid(row=1, column=0, sticky="ew", padx=5)
        ttk.Button(main, text="Clear", command=self.clear).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(main, text="Quit", command=root.destroy).grid(row=1, column=2, sticky="ew", padx=5)

        # Number label
        self.pred_label = ttk.Label(main, text="Prediction: -", font=("Arial", 14))
        self.pred_label.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        # Percentage label
        self.conf_label = ttk.Label(main, text="Confidence: -")
        self.conf_label.grid(row=3, column=0, columnspan=3)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        for c in range(3):
            main.columnconfigure(c, weight=1)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_SIZE
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Prediction: -")
        self.conf_label.config(text="Confidence: -")

    def preprocess(self, pil_img):# crop, padding, resize, normalize, reshape

        # Crop away the blank area
        bbox = pil_img.getbbox()
        if bbox is not None:
            pil_img = pil_img.crop(bbox)

        # Add padding
        pil_img = ImageOps.expand(pil_img, border=20, fill=0)

        # Resize
        pil_img = pil_img.resize((MNIST_SIZE, MNIST_SIZE), Image.Resampling.LANCZOS)

        # normalize
        arr = np.array(pil_img).astype("float32") / 255.0  # (28,28) in [0,1]

        # reshape 28 x 28
        arr = arr.reshape(1, MNIST_SIZE, MNIST_SIZE, 1)
        return arr

    def predict(self):
        x = self.preprocess(self.image)

        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

        self.pred_label.config(text=f"Prediction: {pred}")
        self.conf_label.config(text=f"Confidence: {conf:.2%}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPadApp(root)
    root.mainloop()