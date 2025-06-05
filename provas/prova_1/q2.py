import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils import status_messages


import numpy as np
from scipy.optimize import linprog

# Coeficientes da função objetivo (para maximizar Z, minimizamos -Z)
c = np.array([-140, -300, -400])

# Matriz e vetor para a igualdade de Componente A: 2·x1 + 8·x2 + 2·x3 = 400
A_eq = np.array([[2, 8, 2]])
b_eq = np.array([400])

# Matrizes/vetores das desigualdades:
# 1) Componente B: 1·x1 + 1·x2 + 4·x3 ≤ 200
# 2) Componente C: 1·x1 + 0·x2 + 1·x3 ≤ 300
A_ub = np.array([[1, 1, 4], [1, 0, 1]])
b_ub = np.array([200, 300])

# Limites de não-negatividade: x1, x2, x3 ≥ 0
bounds = [(0, None), (0, None), (0, None)]

# Chamada ao solver
res = linprog(
    c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
)
print("Solução ótima:", res.x)
print("Valor ótimo:", -res.fun)
print("Status:", status_messages[res.status])
