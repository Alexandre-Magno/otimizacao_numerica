import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils import status_messages


import numpy as np
from scipy.optimize import linprog

# 1. Coeficientes da função objetivo (custo por hora de cada inspetor)
c = np.array([5.9, 5.2, 5.5])  # [Pedro, João, Marcelo]

# 2. Restrição de igualdade: x1 + x2 + x3 = 8 (total de horas disponíveis)
A_eq = np.array([[1, 1, 1]])
b_eq = np.array([8])

# 3. Restrições de desigualdade (A_ub @ x ≤ b_ub):
#    a) Volume mínimo: 300x1 + 200x2 + 350x3 ≥ 2000  ⇒  -300x1 -200x2 -350x3 ≤ -2000
#    b) Precisão mínima (peso médio ≥ 98%): 2x2 - 7x3 ≥ 0  ⇒  -2x2 + 7x3 ≤ 0
A_ub = np.array([[-300, -200, -350], [0, -2, 7]])  # volume  # precisão
b_ub = np.array([-2000, 0])

# 4. Bounds de cada variável: 0 ≤ xi ≤ 4 (nenhum inspetor pode trabalhar mais de 4 h)
bounds = [(0, 4), (0, 4), (0, 4)]

# 5. Chamada ao solver
res = linprog(
    c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
)

# 6. Exibição dos resultados
if res.success:
    x1, x2, x3 = res.x
    volume = 300 * x1 + 200 * x2 + 350 * x3
    custo_total = res.fun
    print(
        f"Horas de trabalho (h): Pedro = {x1:.3f}, João = {x2:.3f}, Marcelo = {x3:.3f}"
    )
    print(f"Volume inspecionado: {volume:.1f} cápsulas")
    print(f"Custo diário total: R$ {custo_total:.2f}")
else:
    print("Não foi possível encontrar solução ótima.")
