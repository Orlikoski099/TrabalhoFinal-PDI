import tifffile
import numpy as np
import cv2
import os


def separar_canais_tiff(arquivo_tiff, pasta_saida='canais_separados'):
    nome_base = os.path.splitext(os.path.basename(arquivo_tiff))[0]
    pasta = os.path.join(pasta_saida, nome_base)
    os.makedirs(pasta, exist_ok=True)

    print(f'Carregando {arquivo_tiff} ...')
    img = tifffile.imread(arquivo_tiff)
    print(f"Formato detectado: {img.shape}, dtype={img.dtype}")

    # Caso 1: Imagem 2D (um único canal)
    if img.ndim == 2:
        out_path = os.path.join(pasta, f'{nome_base}_canal_0.png')
        cv2.imwrite(out_path, cv2.convertScaleAbs(img))
        print(f'Canal único salvo em {out_path}')

    # Caso 2: Imagem 3D (multi-página: [n_bandas, H, W])
    elif img.ndim == 3 and img.shape[0] < 30 and img.shape[1] > 32 and img.shape[2] > 32:
        for i in range(img.shape[0]):
            canal = img[i, :, :]
            out_path = os.path.join(pasta, f'{nome_base}_canal_{i}.png')
            cv2.imwrite(out_path, cv2.convertScaleAbs(canal))
            print(f'Canal {i} salvo em {out_path}')

    # Caso 3: Imagem 3D (multicanal: [H, W, n_bandas])
    elif img.ndim == 3 and img.shape[2] < 30 and img.shape[0] > 32 and img.shape[1] > 32:
        for i in range(img.shape[2]):
            canal = img[:, :, i]
            out_path = os.path.join(pasta, f'{nome_base}_canal_{i}.png')
            cv2.imwrite(out_path, cv2.convertScaleAbs(canal))
            print(f'Canal {i} salvo em {out_path}')

    else:
        print('Formato de imagem não suportado automaticamente.')
        print(f'Shape encontrado: {img.shape}')


# ========== USO ==========
if __name__ == '__main__':
    tiff_path = "imgstif\script.tiff"
    separar_canais_tiff(tiff_path)
