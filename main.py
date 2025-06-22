import cv2
import numpy as np
import math
import os

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

def processImg(img):
    caminho_imagem = img
    os.makedirs('resultados', exist_ok=True)
    nome_arquivo = os.path.basename(caminho_imagem)

    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print(f"Erro ao carregar a imagem: {caminho_imagem}")
        return

    altura, largura = imagem.shape[:2]
    limite_tamanho = max(largura, altura) / 3

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(blurred, kernel, iterations=1)

    edges = cv2.Canny(dilated, 50, 150)

    linhas = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=50
    )

    linhas_finais = []

    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            comprimento = distancia_pontos(x1, y1, x2, y2)
            angulo = calcular_angulo(x1, y1, x2, y2)

            if comprimento >= limite_tamanho and (abs(angulo) < 15 or abs(angulo - 90) < 15):
                linhas_finais.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in linhas_finais:
        cv2.line(imagem, (x1, y1), (x2, y2), (0, 0, 255), 2)

    saida = os.path.join('resultados', nome_arquivo)
    cv2.imwrite(saida, imagem)
    print(f"Salvo: {saida}")

    # Visualização
    imagem_resized = cv2.resize(imagem, (1366, 728))
    cv2.imshow(f'Processado - {nome_arquivo}', imagem_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===========================
# MENU INTERATIVO
# ===========================

while True:
    print("\n=== Selecione a imagem para processar ===")
    for idx, nome in enumerate(images):
        print(f"{idx + 1}. {nome}")
    print("0. Processar TODAS as imagens")
    print("-1. Sair")

    try:
        escolha = int(input("\nDigite o número da imagem (ou 0 para todas, -1 para sair): "))

        if escolha == -1:
            print("Encerrando o programa.")
            break

        elif escolha == 0:
            print("\nProcessando todas as imagens...\n")
            for img in images:
                processImg(img)

        elif 1 <= escolha <= len(images):
            img_escolhida = images[escolha - 1]
            print(f"\nProcessando: {img_escolhida}\n")
            processImg(img_escolhida)

        else:
            print("Opção inválida! Tente novamente.")

    except ValueError:
        print("Entrada inválida! Por favor, digite um número.")
