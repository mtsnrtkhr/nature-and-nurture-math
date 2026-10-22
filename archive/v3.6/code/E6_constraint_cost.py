"""E-6 発展2、第11章：実装の制約の値段（部分ごとの学習率、共通の学習率）"""
import numpy as np
from scipy.optimize import minimize
T = 50; R = np.eye(2)
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def loss(H_list, S):                           # 誤りの共分散を追いかけ、部分あたりの平均の誤りを返す
    E, tot = S.copy(), 0.0
    for H in H_list:
        tot += np.trace(E) / 2; A = np.eye(2) - H; E = A @ E @ A.T + H @ R @ H.T
    return tot / T

def ideal(S):                                  # 行列のカルマンフィルタ（理想の学習者）
    P, tot = S.copy(), 0.0
    for _ in range(T):
        tot += np.trace(P) / 2; P = (np.eye(2) - P @ np.linalg.pinv(P + R)) @ P
    return tot / T

x0 = 1 / (np.arange(T) + 2.0)
def best_common(S):                            # 全部の部分に共通の学習率（時間とともに変えてよい）
    return minimize(lambda h: loss([x * np.eye(2) for x in h], S), x0,
                    method="L-BFGS-B", bounds=[(0, 1)] * T).fun
def best_diag(S):                              # 部分ごとの学習率（時間とともに変えてよい）
    return minimize(lambda h: loss([np.diag(h[2*t:2*t+2]) for t in range(T)], S), np.repeat(x0, 2),
                    method="L-BFGS-B", bounds=[(0, 1)] * (2 * T)).fun

for deg in (0, 15, 30, 45):
    S = rot(np.radians(deg)) @ np.diag([1.0, 0.0]) @ rot(np.radians(deg)).T
    i = ideal(S)
    print(f"{deg:2d}度：理想 {i:.3f}、部分ごとの値段 {best_diag(S)-i:.3f}、共通の値段 {best_common(S)-i:.3f}")
for s in (0.1, 0.5, 1.0):
    S = np.diag([1.0, s]); print(f"ばらつき (1, {s})：共通の値段 {best_common(S)-ideal(S):.4f}")
# 一つ変えてみる：np.diag([1.0, 0.0]) を np.diag([1.0, 0.3]) にして、回転したときの値段を見る
