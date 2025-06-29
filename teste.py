import tifffile
import numpy as np
import cv2
import os

# 1. Carregar a imagem TIFF colorida
tiff_path = 'imgstif\script.tiff'  # Use o caminho do seu TIFF se necessário
if tiff_path.lower().endswith('.tif') or tiff_path.lower().endswith('.tiff'):
    img = tifffile.imread(tiff_path)
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    if img.shape[-1] == 1:
        img = img[:, :, 0]
else:
    img = cv2.imread(tiff_path)

# 2. Converter para escala de cinza se for colorida
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img

# 3. Realçar contraste (opcional, mas geralmente ajuda)
gray = cv2.equalizeHist(gray)

# 4. Binarizar automaticamente (Otsu)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Filtros morfológicos para pegar linhas horizontais e verticais (grid)
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))  # Horizontal
horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)

kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))  # Vertical
vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)

# 6. Combinar os dois para o grid completo
grid_mask = cv2.bitwise_or(horizontal, vertical)

# 7. (Opcional) HoughLinesP para deixar só as linhas retas e longas
lines = cv2.HoughLinesP(grid_mask, 1, np.pi/180,
                        threshold=100, minLineLength=200, maxLineGap=30)
mask_lines = np.zeros_like(gray)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), 255, 2)

# 8. Salvar resultados
os.makedirs('resultados', exist_ok=True)
cv2.imwrite('resultados/grid_ruas_binaria.png', grid_mask)
cv2.imwrite('resultados/grid_ruas_hough.png', mask_lines)

print('Máscaras salvas em: resultados/grid_ruas_binaria.png e resultados/grid_ruas_hough.png')
