"""E-1 第3章 図3-1：カルマンの利得の下がり方（観測雑音 r = 1 のスカラー）"""
import numpy as np

def gain(P0, q, r=1.0, T=400):
    P, ks = P0, []
    for _ in range(T):
        K = P / (P + r); ks.append(K); P = (1 - K) * P + q
    return np.array(ks)

for P0, q, name in [(10, 0, "書かれていない部分・動かない目標"),
                    (0.05, 0, "書かれている部分"),
                    (10, 0.01, "動く目標")]:
    k = gain(P0, q)
    print(f"{name}: K_1={k[0]:.3f}  K_10={k[9]:.4f}  K_100={k[99]:.4f}  K_400={k[399]:.4f}")
print("動く目標の近似 sqrt(q/r) =", np.sqrt(0.01))
# 一つ変えてみる：q を 0.001, 0.1 にして、落ち着く値が sqrt(q/r) に近いかを確かめる
