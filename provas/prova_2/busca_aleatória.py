import random
import math


def funcao_objetivo(x):
    """
    Função objetivo: cos(x) * tan(x) - x**2
    """
    return math.cos(x[0]) * math.tan(x[0]) - x[0] ** 2


def busca_aleatoria(objetivo, espaco_busca, n_iteracoes):
    """
    Executa o algoritmo de busca aleatória para minimização.

    Args:
        objetivo (function): A função a ser minimizada.
        espaco_busca (list): Uma lista de tuplas, onde cada tupla define o
                             intervalo [min, max] para cada dimensão da busca.
        n_iteracoes (int): O número total de iterações (pontos a serem testados).

    Returns:
        tuple: Uma tupla contendo a melhor solução encontrada (lista de coordenadas)
               e o valor correspondente da função objetivo.
    """
    # Inicializa a melhor solução encontrada com None e a melhor avaliação com infinito
    melhor_solucao = None
    melhor_avaliacao = float("inf")

    print(f"Iniciando Busca Aleatória por {n_iteracoes} iterações...")

    for i in range(n_iteracoes):
        # 1. Gera uma solução candidata aleatória dentro do espaço de busca
        solucao_candidata = [
            random.uniform(espaco_busca[d][0], espaco_busca[d][1])
            for d in range(len(espaco_busca))
        ]

        # 2. Avalia a solução candidata usando a função objetivo
        avaliacao_candidata = objetivo(solucao_candidata)

        # 3. Se a candidata for melhor, atualiza a melhor solução encontrada
        if avaliacao_candidata < melhor_avaliacao:
            melhor_solucao = solucao_candidata
            melhor_avaliacao = avaliacao_candidata
            print(
                f"> Iteração {i+1:4d}, Nova Melhor Avaliação = {melhor_avaliacao:.6f}"
            )

    return melhor_solucao, melhor_avaliacao


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    # 1. Definir o espaço de busca (limites para cada variável)
    # Para a nossa função de 2 variáveis (x, y), definimos os limites de -100 a 100 para cada.
    # Formato: [[min_x, max_x], [min_y, max_y], ...]
    espaco_busca = [[-10.0, 10.0]]

    # 2. Definir o número total de iterações
    n_iteracoes = 1000

    # 3. Executar a busca aleatória
    melhor_solucao, melhor_avaliacao = busca_aleatoria(
        funcao_objetivo, espaco_busca, n_iteracoes
    )

    # 4. Apresentar os resultados finais
    print("\n--- Busca Aleatória Concluída! ---")
    if melhor_solucao:
        solucao_formatada = [f"{coord:.6f}" for coord in melhor_solucao]
        print(f"Melhor Solução Encontrada (x, y): {solucao_formatada}")
        print(f"Valor da Função Objetivo na Melhor Solução: {melhor_avaliacao:.6f}")
    else:
        print("Nenhuma solução foi encontrada.")
