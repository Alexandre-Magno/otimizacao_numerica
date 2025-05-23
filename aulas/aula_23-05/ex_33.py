from scipy.optimize import linprog

# Coeficientes de horas por unidade produzida
# [m1, m2, m3]
montagem = [0.1, 0.3, 0.4]    # horas de montagem
config = [0.2, 0.1, 0.1]      # horas de configuração
verificacao = [0.1, 0.1, 0.0]  # horas de verificação

# Limites de horas disponíveis
horas_disponiveis = {
    'montagem': 290,
    'config': 250,
    'verificacao': 110
}

# Lucro unitário (objetivo é maximizar, então usamos o negativo para minimizar)
lucro_unitario = [-100, -210, -250]  # negativo porque queremos maximizar

# Restrições:
# 1. qtd_m1 >= 1
# 2. qtd_m2 >= 1
# 3. qtd_m3 >= 1
# 4. qtd_m3 >= 2 * qtd_m2  ->  -qtd_m3 + 2*qtd_m2 <= 0
# 5. Restrições de horas

# Coeficientes das restrições de desigualdade (A_ub * x <= b_ub)
A_ub = [
    # Restrição qtd_m3 >= 2*qtd_m2 (reescrita como -qtd_m3 + 2*qtd_m2 <= 0)
    [0, 2, -1],
    
    # Restrições de horas (<=)
    montagem,       # horas de montagem
    config,         # horas de configuração
    verificacao,    # horas de verificação
    
    # Restrições de não-negatividade (x_i >= 1)
    [-1, 0, 0],    # -qtd_m1 <= -1
    [0, -1, 0],     # -qtd_m2 <= -1
    [0, 0, -1]      # -qtd_m3 <= -1
]

# Lado direito das desigualdades
b_ub = [
    0,  # para a restrição qtd_m3 >= 2*qtd_m2
    horas_disponiveis['montagem'],
    horas_disponiveis['config'],
    horas_disponiveis['verificacao'],
    -1,  # para qtd_m1 >= 1
    -120,  # para qtd_m2 >= 1
    -1   # para qtd_m3 >= 1
]

# Limites para as variáveis (todas >= 0, sem limite superior)
bounds = [(1, None), (120, None), (1, None)]

# Resolvendo o problema de maximização (por isso usamos o negativo do lucro)
res = linprog(
    c=lucro_unitario,  # coeficientes da função objetivo
    A_ub=A_ub,         # coeficientes das restrições de desigualdade
    b_ub=b_ub,         # lados direitos das desigualdades
    bounds=bounds,     # limites das variáveis
    method='highs'     # método recomendado para problemas de programação linear
)

# Exibindo os resultados
if res.success:
    print("Solução ótima encontrada:")
    print(f"Quantidade do Modelo 1: {res.x[0]:.2f} unidades")
    print(f"Quantidade do Modelo 2: {res.x[1]:.2f} unidades")
    print(f"Quantidade do Modelo 3: {res.x[2]:.2f} unidades")
    print(f"Lucro total: R$ {-res.fun:.2f}")
    
    # Verificando se qtd_m3 >= 2*qtd_m2
    qtd_m2 = res.x[1]
    qtd_m3 = res.x[2]
    print(f"\nVerificação da restrição qtd_m3 >= 2*qtd_m2: {qtd_m3} >= {2*qtd_m2}? {qtd_m3 >= 2*qtd_m2}")
else:
    print("Otimização não foi bem-sucedida.")
    print("Status:", res.message)