import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import cv2
import numpy as np

# === Load dataset warna ===
dataset = pd.read_csv("DATASET_COLORS_RHEHAN_ADI_AUGMENTED.csv", sep=";")

# Ambil kolom penting
color_data = dataset[['ColorName', 'R', 'G', 'B']].copy()
color_data['R'] = pd.to_numeric(color_data['R'], errors='coerce')
color_data['G'] = pd.to_numeric(color_data['G'], errors='coerce')
color_data['B'] = pd.to_numeric(color_data['B'], errors='coerce')
color_data = color_data.dropna()

# === Pisahkan fitur dan label ===
X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# === Normalisasi data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Latih model Decision Tree pakai semua data ===
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_scaled, y)

# === Deteksi warna via kamera ===
cap = cv2.VideoCapture(0)

# tampil style
text_x = 20
start_y = 40
line_height = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2   # titik tengah
    box_size = 25             # half size -> kotak 50x50 px

    # preview box size di ujung kanan
    preview_w, preview_h = 80, 30
    margin = 16

    # tentukan 3 titik pengukuran (kiri, tengah, kanan)
    points = [
        (w // 4, cy),
        (cx, cy),
        (3 * w // 4, cy),
    ]

    results = []  # simpan hasil deteksi untuk ditampilkan rapi

    for px, py in points:
        # koordinat kotak (pastikan ter-clamp ke frame)
        x1 = max(0, px - box_size)
        y1 = max(0, py - box_size)
        x2 = min(w, px + box_size)
        y2 = min(h, py + box_size)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            avg_bgr = np.array([0, 0, 0])
        else:
            avg_bgr = np.mean(roi, axis=(0, 1))
        b, g, r = [int(c) for c in avg_bgr]

        # Normalisasi pixel rata-rata (OpenCV BGR → RGB)
        pixel_rgb = [r, g, b]
        pixel_scaled = scaler.transform([pixel_rgb])

        # Prediksi warna + accuracy pakai Decision Tree
        probs = dt.predict_proba(pixel_scaled)[0]
        color_pred = dt.classes_[np.argmax(probs)]
        accuracy = np.max(probs) * 100

        # Simpan hasil
        results.append((color_pred, accuracy, (b, g, r), (x1, y1, x2, y2), (px, py)))

    # gambar kotak pengukuran di frame (kiri/center/kanan)
    for (_, _, (b, g, r), (x1, y1, x2, y2), (px, py)) in results:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
        cv2.circle(frame, (px, py), 4, (0, 0, 0), -1)

    # gambar background semi-solid untuk teks
    preview_x1 = w - margin - preview_w
    text_bg_x2 = max(preview_x1 - 12, text_x + 100)
    bg_y1 = start_y - 28
    bg_y2 = start_y + line_height * len(results) - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 8, bg_y1), (text_bg_x2, bg_y2), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # tampilkan teks + preview box
    for i, (color_pred, accuracy, (b, g, r), _, _) in enumerate(results):
        y_offset = start_y + i * line_height
        text = f"Color {i+1}: {color_pred} ({accuracy:.1f}%)"
        cv2.putText(frame, text, (text_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        px1 = preview_x1
        py1 = y_offset - 20
        px2 = px1 + preview_w
        py2 = py1 + preview_h

        px1 = int(max(margin, px1))
        px2 = int(min(w - margin, px2))
        py1 = int(max(8, py1))
        py2 = int(min(h - 8, py2))

        cv2.rectangle(frame, (px1, py1), (px2, py2), (int(b), int(g), int(r)), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
