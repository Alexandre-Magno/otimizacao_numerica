from scipy.optimize import linprog
import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import status_messages

# função custo
c = [7, 4, -15]


# variaveis restrições
A = [[1 / 3, -32 / 9, 20 / 9], [1 / 6, -13 / 9, 5 / 18], [-2 / 3, 16 / 9, -1 / 9]]

# lado direito das restrições
b = [1, 2, 3]

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
