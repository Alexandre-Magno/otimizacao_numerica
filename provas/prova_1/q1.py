import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils import status_messages

import numpy as np
from scipy.optimize import linprog


# Vetor de custos (função-objetivo): [x1, x2, y1, y2]
c = np.array([10, 12, 8, 10])

# Restrições convertidas para A_ub * x ≤ b_ub
A_ub = np.array(
    [
        [-1, -1, 0, 0],  # x1 + x2 ≥ 400  →  -x1 - x2 ≤ -400
        [0, 0, -1, -1],  # y1 + y2 ≥ 200  →  -y1 - y2 ≤ -200
        [1, -1, 1, -1],  # x2 + y2 ≥ x1 + y1  →  x1 - x2 + y1 - y2 ≤ 0
        [-1, -1, -1, -1],  # x1 + x2 + y1 + y2 ≥ 1000  →  -(…) ≤ -1000
    ]
)
b_ub = np.array([-400, -200, 0, -1000])

# Limites de cada variável: todas ≥ 0
bounds = [(0, None)] * 4

# Resolução
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

print("Solução ótima:", res.x)
print("Valor ótimo:", -res.fun)
print("Status:", status_messages[res.status])
