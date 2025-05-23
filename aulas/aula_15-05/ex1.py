from scipy.optimize import linprog

import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import status_messages


# Coeficientes da função objetivo (negativos para maximização)
# Max Z = 3x1 − 5x2
c = [-3, 5]

# Coeficientes das restrições (lado esquerdo)
# x1 ≤ 4
# 2x2 ≤ 12
# 3x1 + 2x2 ≥ 18

A = [[1, 0], [0, 2], [-3, -2]]

# Lado direito das restrições
b = [4, 12, -18]

# Restrições de não negatividade são padrão em linprog
res = linprog(
    c,
    A_ub=A,
    b_ub=b,
    method="highs",
)

print("Solução ótima:", res.x)
print("Valor ótimo:", -res.fun)
print("Status:", status_messages[res.status])
