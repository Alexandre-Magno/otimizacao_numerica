from scipy.optimize import linprog

import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import status_messages


# Coeficientes da função objetivo (negativos para maximização)
c = [5, -3, -8]

# Coeficientes das restrições (lado esquerdo)
A = [[2, 5, -1], [-2, -12, 3], [-3, -8, 2]]

# Lado direito das restrições
b = [1, 9, 4]

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
