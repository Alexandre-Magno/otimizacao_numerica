import numpy as np
import matplotlib.pyplot as plt


def funcao_objetivo_rosenbrock(p):
    """
    Função de Rosenbrock. É um desafio clássico para algoritmos de otimização.
    O mínimo global é 0, localizado no ponto (1, 1).
    Recebe um vetor p, onde p[0] é x e p[1] é y.
    """
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def line_search(func, p, direction, tol=1e-5, bounds=(-1.0, 1.0)):
    """
    Busca ternária para encontrar o mínimo de uma função ao longo de uma direção.
    Encontra o escalar 'lambda' que minimiza func(p + lambda * direction).
    """

    # Função 1D para a busca
    def f_1d(lam):
        return func(p + lam * direction)

    low, high = bounds

    # A busca ternária é um método simples para encontrar o mínimo em um intervalo.
    while (high - low) > tol:
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        if f_1d(m1) < f_1d(m2):
            high = m2
        else:
            low = m1

    return (low + high) / 2


def powell(func, x0, tol=1e-6, max_iter=100, history=False):
    """
    Executa o Método de Powell para minimização de funções sem uso de derivadas.

    Args:
        func (function): A função a ser minimizada.
        x0 (list or np.array): O ponto de partida inicial.
        tol (float): A tolerância para o critério de parada.
        max_iter (int): O número máximo de iterações.

    Returns:
        tuple: Uma tupla contendo a melhor solução (np.array),
               o melhor valor da função, e o histórico de pontos
               visitados (lista de np.array).
    """
    p = np.array(x0, dtype=float)
    n_dims = len(p)
    # Inicializa o conjunto de direções com a base canônica (matriz identidade)
    directions = np.identity(n_dims)
    f_val = func(p)

    path_history = [p.copy()]
    print(f"Iniciando Método de Powell em x0={p} com f(x0)={f_val:.6f}")

    for i in range(max_iter):
        p_prev = p.copy()
        f_prev = f_val

        biggest_drop = 0.0
        drop_index = 0

        # 1. Realiza buscas lineares ao longo de cada direção no conjunto
        for j in range(n_dims):
            d = directions[j]
            f_before_ls = func(p)

            # Encontra o passo ótimo lambda para a busca linear
            lambda_min = line_search(func, p, d)
            p += lambda_min * d

            # Guarda qual direção causou a maior queda no valor da função
            drop = f_before_ls - func(p)
            if drop > biggest_drop:
                biggest_drop = drop
                drop_index = j
        path_history.append(p.copy())

        f_val = func(p)

        # 2. Critério de convergência
        # Se a melhora for muito pequena, paramos.
        if 2 * abs(f_prev - f_val) <= tol * (abs(f_prev) + abs(f_val) + 1e-8):
            print(f"\nConvergência atingida na iteração {i+1}.")
            break

        # 3. Calcula a nova direção e avalia a condição de Powell/Brent
        new_dir = p - p_prev
        f_extrapolated = func(p + new_dir)

        # Esta condição complexa evita que o conjunto de direções se torne
        # linearmente dependente, o que paralisaria o algoritmo.
        if f_extrapolated < f_prev:
            term1 = (f_prev - f_val - biggest_drop) ** 2
            term2 = (f_prev - f_extrapolated) ** 2
            if 2 * (f_prev - 2 * f_val + f_extrapolated) * term1 < biggest_drop * term2:
                # Se a condição for satisfeita, fazemos uma busca na nova direção
                # e a substituímos no conjunto de direções.
                lambda_min = line_search(func, p, new_dir)
                p = p + lambda_min * new_dir
                path_history.append(p.copy())

                # Normaliza e substitui a direção que teve a maior queda
                directions[drop_index] = new_dir / np.linalg.norm(new_dir)

        print(f"> Iteração {i+1:3d}: f(x) = {func(p):.8f}")
    return p, func(p), np.array(path_history)


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    # 1. Ponto de partida inicial
    # Um ponto inicial comum para a função de Rosenbrock
    ponto_inicial = [-1.0, 2.0]

    # 2. Executar o método de Powell
    melhor_solucao, melhor_avaliacao, historico = powell(
        funcao_objetivo_rosenbrock, ponto_inicial, max_iter=50
    )

    # 3. Apresentar os resultados finais
    print("\n--- Método de Powell Concluído! ---")
    solucao_formatada = [f"{coord:.6f}" for coord in melhor_solucao]
    print(f"Ponto de Mínimo Encontrado (x, y): {solucao_formatada}")
    print(f"Valor da Função no Mínimo: {melhor_avaliacao:.8f}")
    print("(Valor teórico esperado: 0.0 no ponto [1.0, 1.0])")

    # 4. Gerar o gráfico do caminho da otimização
    print("\nGerando o gráfico do caminho da otimização...")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Curvas de nível da função de Rosenbrock
    x_range = np.linspace(-2.0, 2.0, 250)
    y_range = np.linspace(-1.0, 3.0, 250)
    X, Y = np.meshgrid(x_range, y_range)
    Z = funcao_objetivo_rosenbrock([X, Y])

    contour = ax.contour(X, Y, Z, levels=np.logspace(0, 3.5, 25), cmap="viridis_r")
    fig.colorbar(contour, ax=ax, label="Valor da Função Objetivo (escala log)")

    # Plot do caminho da otimização
    path_x = historico[:, 0]
    path_y = historico[:, 1]
    ax.plot(
        path_x, path_y, "r-o", markersize=4, linewidth=1.5, label="Caminho de Powell"
    )

    # Ponto inicial, final e mínimo teórico
    ax.plot(
        ponto_inicial[0], ponto_inicial[1], "go", markersize=10, label="Ponto Inicial"
    )
    ax.plot(
        melhor_solucao[0],
        melhor_solucao[1],
        "b*",
        markersize=15,
        label="Ponto Final (Mínimo)",
    )
    ax.plot(1, 1, "m+", markersize=15, markeredgewidth=3, label="Mínimo Global Teórico")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Método de Powell na Função de Rosenbrock")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # 5. Gerar o gráfico 3D do caminho da otimização
    print("\nGerando o gráfico 3D do caminho da otimização...")
    fig_3d = plt.figure(figsize=(12, 9))
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    # A malha para o gráfico de superfície já foi calculada para o gráfico 2D
    # X, Y, Z

    # Plot da superfície da função de Rosenbrock
    ax_3d.plot_surface(
        X, Y, Z, cmap="viridis_r", alpha=0.6, rcount=100, ccount=100, edgecolor="none"
    )

    # Plot do caminho da otimização em 3D
    path_x = historico[:, 0]
    path_y = historico[:, 1]
    path_z = funcao_objetivo_rosenbrock([path_x, path_y])
    ax_3d.plot(
        path_x,
        path_y,
        path_z,
        "r-o",
        markersize=5,
        linewidth=2,
        label="Caminho de Powell",
    )

    # Ponto inicial, final e mínimo teórico em 3D
    ax_3d.scatter(
        ponto_inicial[0],
        ponto_inicial[1],
        funcao_objetivo_rosenbrock(ponto_inicial),
        c="g",
        s=100,
        label="Ponto Inicial",
        depthshade=True,
    )
    ax_3d.scatter(
        melhor_solucao[0],
        melhor_solucao[1],
        melhor_avaliacao,
        c="b",
        marker="*",
        s=200,
        label="Ponto Final (Mínimo)",
        depthshade=True,
    )
    ax_3d.scatter(
        1,
        1,
        0,
        c="m",
        marker="+",
        s=200,
        label="Mínimo Global Teórico",
        depthshade=True,
    )

    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("f(x, y)")
    ax_3d.set_title("Método de Powell na Função de Rosenbrock (3D)")
    ax_3d.legend()
    plt.show()
