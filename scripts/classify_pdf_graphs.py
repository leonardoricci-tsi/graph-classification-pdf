import fitz 
import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
PDF_PATH = "input/seu_arquivo.pdf"
OUTPUT_DIR = "output/gráficos_recortados"
MODEL_PATH = "model/best.pt"
IMAGE_SIZE = 224  # mesma usada no treino

#Cria diretórios 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega modelo YOLOv8n-cls
model = YOLO(MODEL_PATH)

# salvar imagens 
def save_image(image, path):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# 1. Extrair imagens das páginas 
doc = fitz.open(PDF_PATH)
img_counter = 0

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)  # renderiza a página como imagem
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # remove canal alpha se houver
        img = img[:, :, :3]

    # 2. Pré-processa imagem para detectar regiões 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:  
            cropped = img[y:y+h, x:x+w]
            resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

            # 3. Classifica com o modelo
            results = model(resized, verbose=False)
            prediction = results[0]
            label = prediction.names[prediction.probs.top1]
            conf = prediction.probs.top1conf.item()

            # 4. Salva imagem com classe no nome 
            output_path = os.path.join(OUTPUT_DIR, f"pag{i+1}_graf{img_counter+1}_{label}_{conf:.2f}.png")
            save_image(cropped, output_path)

            
            print(f"[Página {i+1}] Gráfico detectado: {label} ({conf:.2f})")
            img_counter += 1

print(f"\n✅ Total de gráficos classificados: {img_counter}")
