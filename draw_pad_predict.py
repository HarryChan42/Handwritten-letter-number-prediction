"""
draw_pad_predict_emnist.py

Tkinter draw pad for EMNIST model prediction (letters OR byclass).
Features:
- Big drawing canvas (black background, white ink)
- Live 28x28 preview
- Live prediction every ~150ms
- Top-K probabilities
- Clear button
- Optional orientation fix + invert to match your training pipeline

Install:
  pip install pillow tensorflow numpy

Run:
  python draw_pad_predict_emnist.py

Make sure MODEL_PATH points to your trained .keras model.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps, ImageTk


# -----------------------
# Config (EDIT THESE)
# -----------------------
MODEL_PATH = "emnist_byclass_cnn.keras"  # or "emnist_letters_cnn.keras"
CLASS_NAMES_PATH = "class_names.txt"     # same one you saved during training

CANVAS_SIZE = 280     # 10x scale of 28
BRUSH_SIZE = 18
INPUT_SIZE = 28

PRED_INTERVAL_MS = 150
MIN_INK_SUM = 30      # ignore prediction when canvas is basically empty

# Your training preprocessing used:
# image = rot90(k=3) then flip_left_right
# Only enable if your drawpad predictions are rotated/consistently wrong.
APPLY_ROT90_K3 = False
APPLY_FLIP_LR = False

# Training used: image = tf.cast(image)/255 and kept white strokes on black background.
# If your training data expects black ink on white background, set INVERT_COLORS=True.
INVERT_COLORS = False

TOP_K = 5


# -----------------------
# Helpers
# -----------------------
def load_class_names(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip() != ""]
        return names
    except FileNotFoundError:
        return None


def pil_to_model_input(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL grayscale image into a (1, 28, 28, 1) float32 tensor in [0,1]
    with optional transforms to match training.
    """
    img = pil_img.copy()

    # Optional invert
    if INVERT_COLORS:
        img = ImageOps.invert(img)

    # Optional orientation fix (ONLY if needed)
    if APPLY_ROT90_K3:
        img = img.rotate(-90, expand=False)  # rot90 k=3 == rotate clockwise 90
    if APPLY_FLIP_LR:
        img = ImageOps.mirror(img)

    # Resize to model input
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(img).astype("float32") / 255.0  # (28,28), 0..1
    arr = arr[..., np.newaxis]                     # (28,28,1)
    arr = np.expand_dims(arr, 0)                   # (1,28,28,1)
    return arr


def center_and_scale(pil_img: Image.Image) -> Image.Image:
    """
    Improve predictions by cropping to the ink bounding box and centering it.
    Keeps background black and ink white (by default).
    """
    img = pil_img.copy()

    # Find bounding box of non-black pixels
    arr = np.array(img)
    ys, xs = np.where(arr > 10)
    if len(xs) == 0 or len(ys) == 0:
        return img

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Add padding around the ink
    pad = 20
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1] - 1, x1 + pad)
    y1 = min(arr.shape[0] - 1, y1 + pad)

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Paste into square canvas to preserve aspect ratio
    w, h = cropped.size
    side = max(w, h)
    square = Image.new("L", (side, side), color=0)  # black background
    square.paste(cropped, ((side - w) // 2, (side - h) // 2))

    return square


# -----------------------
# App
# -----------------------
class EMNISTPadApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMNIST Draw Pad (Live Prediction)")

        # Load model
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Load class names if available
        self.class_names = load_class_names(CLASS_NAMES_PATH)
        if self.class_names is None:
            # fallback: numeric labels
            out_dim = self.model.output_shape[-1]
            self.class_names = [str(i) for i in range(out_dim)]

        # Layout
        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Left: drawing + preview
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Backing image for canvas
        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)  # black
        self.draw = ImageDraw.Draw(self.pil_img)

        # Brush binds
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Controls row
        ctrl = ttk.Frame(left)
        ctrl.grid(row=1, column=0, pady=(5, 0), sticky="ew")

        self.clear_btn = ttk.Button(ctrl, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=0, padx=5)

        ttk.Label(ctrl, text="Brush").grid(row=0, column=1, padx=(10, 2))
        self.brush_var = tk.IntVar(value=BRUSH_SIZE)
        self.brush_slider = ttk.Scale(ctrl, from_=6, to=40, variable=self.brush_var, orient="horizontal")
        self.brush_slider.grid(row=0, column=2, padx=5, sticky="ew")
        ctrl.columnconfigure(2, weight=1)

        # Preview (28x28 scaled up)
        preview_frame = ttk.LabelFrame(left, text="28×28 Preview", padding=5)
        preview_frame.grid(row=2, column=0, pady=10, sticky="ew")

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0)

        # Right: prediction panel
        right = ttk.Frame(main, padding=(15, 0, 0, 0))
        right.grid(row=0, column=1, sticky="n")

        self.pred_title = ttk.Label(right, text="Prediction", font=("Segoe UI", 14, "bold"))
        self.pred_title.grid(row=0, column=0, sticky="w")

        self.pred_main = ttk.Label(right, text="—", font=("Segoe UI", 28, "bold"))
        self.pred_main.grid(row=1, column=0, sticky="w", pady=(5, 10))

        self.pred_conf = ttk.Label(right, text="Confidence: —", font=("Segoe UI", 11))
        self.pred_conf.grid(row=2, column=0, sticky="w")

        ttk.Separator(right, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=10)

        self.topk_title = ttk.Label(right, text=f"Top {TOP_K}", font=("Segoe UI", 12, "bold"))
        self.topk_title.grid(row=4, column=0, sticky="w")

        self.topk_box = tk.Text(right, width=28, height=10, font=("Consolas", 11))
        self.topk_box.grid(row=5, column=0, sticky="w")
        self.topk_box.configure(state="disabled")

        # Kick off periodic prediction
        self.root.after(PRED_INTERVAL_MS, self.predict_loop)

    def paint(self, event):
        r = int(self.brush_var.get())
        x, y = event.x, event.y
        x0, y0, x1, y1 = x - r, y - r, x + r, y + r

        # Draw on Tk canvas (white)
        self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        # Draw on PIL image
        self.draw.ellipse([x0, y0, x1, y1], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.pil_img)
        self.pred_main.config(text="—")
        self.pred_conf.config(text="Confidence: —")
        self._set_topk_text("")

    def _set_topk_text(self, text: str):
        self.topk_box.configure(state="normal")
        self.topk_box.delete("1.0", tk.END)
        self.topk_box.insert(tk.END, text)
        self.topk_box.configure(state="disabled")

    def predict_loop(self):
        # Skip if empty
        ink_sum = np.array(self.pil_img).sum()
        if ink_sum < MIN_INK_SUM * 255:
            # Still update preview to show empty
            self._update_preview(self.pil_img)
            self.root.after(PRED_INTERVAL_MS, self.predict_loop)
            return

        # Center/crop to ink area (often improves accuracy a lot)
        processed = center_and_scale(self.pil_img)

        # Update preview (after centering, before resize)
        self._update_preview(processed)

        # Convert to model input
        x = pil_to_model_input(processed)

        # Predict
        probs = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else str(pred_idx)
        conf = float(probs[pred_idx])

        # Top-K
        topk_idx = np.argsort(probs)[::-1][:TOP_K]
        lines = []
        for i in topk_idx:
            name = self.class_names[int(i)] if int(i) < len(self.class_names) else str(int(i))
            lines.append(f"{name:>3}  {probs[int(i)]*100:6.2f}%")
        topk_text = "\n".join(lines)

        # Update UI
        self.pred_main.config(text=pred_name)
        self.pred_conf.config(text=f"Confidence: {conf*100:.2f}%")
        self._set_topk_text(topk_text)

        # Schedule next run
        self.root.after(PRED_INTERVAL_MS, self.predict_loop)

    def _update_preview(self, pil_img: Image.Image):
        # Make a visible preview of the 28x28 input (scaled up)
        img28 = pil_img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)

        # Apply same transforms as input so preview matches what model "sees"
        if INVERT_COLORS:
            img28 = ImageOps.invert(img28)
        if APPLY_ROT90_K3:
            img28 = img28.rotate(-90, expand=False)
        if APPLY_FLIP_LR:
            img28 = ImageOps.mirror(img28)

        # Scale up for UI
        scale = 6
        shown = img28.resize((INPUT_SIZE * scale, INPUT_SIZE * scale), Image.Resampling.NEAREST)
        self._preview_tk = ImageTk.PhotoImage(shown)
        self.preview_label.configure(image=self._preview_tk)


def main():
    root = tk.Tk()
    # Use ttk theme if available
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = EMNISTPadApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
