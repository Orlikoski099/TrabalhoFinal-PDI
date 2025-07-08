import cv2
import numpy as np
import math
import os
import threading

images = [
    './ImgsGE/1 - curitiba.jpg',
    './ImgsGE/2 - rio de janeiro.jpg',
    './ImgsGE/3 - sao jorge doeste.jpg',
    './ImgsGE/4 - barcelona.jpg',
    './ImgsGE/5 - new dheli.jpg',
    './ImgsGE/6 - ancara.jpg',
    './ImgsGE/7 - tokyo.jpg',
    './ImgsGE/8 - san francisco.jpg',
    './ImgsGE/9 - las vegas.jpg',
    './ImgsGE/10 - brasilia.jpg',
    './ImgsGE/11 - tres lagoas.jpg',
    './ImgsGE/12 - Inazawa.jpg',
]

def calcular_angulo(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def distancia_pontos(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def processImg(img, metodo):
    caminho_imagem = img
    os.makedirs('resultados', exist_ok=True)
    nome_arquivo = os.path.basename(caminho_imagem)

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {caminho_imagem}")
        return

    saida_img = imagem.copy()

    if metodo == 1:
        altura, largura = imagem.shape[:2]
        limite_tamanho = max(largura, altura) / 3

        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(blurred, kernel, iterations=1)
        edges = cv2.Canny(dilated, 50, 150)

        linhas = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                 threshold=80, minLineLength=150, maxLineGap=30)

        if linhas is not None:
            for linha in linhas:
                x1, y1, x2, y2 = linha[0]
                comprimento = distancia_pontos(x1, y1, x2, y2)
                angulo = calcular_angulo(x1, y1, x2, y2)
                if comprimento >= limite_tamanho and (abs(angulo) < 15 or abs(angulo - 90) < 15):
                    cv2.line(saida_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    elif metodo == 2:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 60])
        upper = np.array([180, 30, 180])
        mascara = cv2.inRange(hsv, lower, upper)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        saida_img[mascara > 0] = [0, 0, 255]

    elif metodo == 3:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        dilated = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=1)

        contornos, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h != 0 else 0

            if area > 300 and (aspect_ratio > 2 or aspect_ratio < 0.5):
                cv2.drawContours(saida_img, [cnt], -1, (0, 0, 255), 2)

    elif metodo == 4:
        hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 60])
        upper = np.array([180, 30, 180])
        mascara = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=2)

        # (Opcional) Remove áreas pequenas
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        min_area = 300  # ajuste se necessário
        mascara_filtrada = np.zeros_like(mascara)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                mascara_filtrada[labels == i] = 255
        # Se não quiser remover nada, só use "mascara_filtrada = mascara"

        # Pinta as ruas de vermelho, mantém o resto da imagem igual
        saida_img[mascara_filtrada > 0] = [0, 0, 255]

    else:
        print("Método inválido!")
        return

    saida = os.path.join('resultados', f"{metodo}_{nome_arquivo}")
    cv2.imwrite(saida, saida_img)
    print(f"Salvo: {saida}")


    # imagem_resized = cv2.resize(saida_img, (1366, 728))
    # cv2.imshow(f'Processado ({metodo}) - {nome_arquivo}', imagem_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# ===========================
# MENU INTERATIVO COM THREADS
# ===========================

def start_thread(img_path, metodo):
    thread = threading.Thread(target=processImg, args=(img_path, metodo))
    thread.start()
    return thread

while True:
    print("\n=== Selecione a imagem para processar ===")
    for idx, nome in enumerate(images):
        print(f"{idx + 1}. {os.path.basename(nome)}")
    print("0. Processar TODAS as imagens")
    print("-1. Sair")

    try:
        escolha = int(input("\nDigite o número da imagem (ou 0 para todas, -1 para sair): "))

        if escolha == -1:
            print("Encerrando o programa.")
            break

        elif escolha == 0 or (1 <= escolha <= len(images)):
            print("\n=== Selecione o método de processamento ===")
            print("1. Transformada de Hough (linhas retas)")
            print("2. Segmentação por cor (HSV)")
            print("3. Contornos + morfologia")
            print("4. HSV + Hough")
            metodo = int(input("Escolha o método (1, 2, 3 ou 4): "))

            if metodo not in [1, 2, 3, 4]:
                print("Método inválido. Tente novamente.")
                continue

            threads = []

            if escolha == 0:
                for img in images:
                    t = start_thread(img, metodo)
                    threads.append(t)
                # Espera todas as threads terminarem (opcional)
                for t in threads:
                    t.join()
            else:
                img_escolhida = images[escolha - 1]
                start_thread(img_escolhida, metodo)

        else:
            print("Opção inválida! Tente novamente.")

    except ValueError:
        print("Entrada inválida! Por favor, digite um número.")
