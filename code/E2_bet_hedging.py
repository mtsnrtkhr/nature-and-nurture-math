"""E-2 第6章 図6-1：平均で測るか、増殖率で測るか"""
import numpy as np
p_good, mult = 0.7, 3.0          # 良い年の確率、芽を出した種の倍率（悪い年は全滅）
g = np.linspace(0, 0.999, 1000)  # 芽を出させる割合
arith = p_good * (1 - g + mult * g) + (1 - p_good) * (1 - g)
growth = p_good * np.log(1 - g + mult * g) + (1 - p_good) * np.log(1 - g)
i = np.argmax(growth)
print(f"算術平均の最大：g = 1 で {p_good*mult:.2f} 倍")
print(f"増殖率の最大：g* = {g[i]:.3f}, Lambda = {growth[i]:.3f}, 1年あたり {np.exp(growth[i]):.2f} 倍")
# 一つ変えてみる：悪い年の倍率を 0 から 0.2 に上げる（芽を出した種の一部が生き残る）と g* はどう動くか
