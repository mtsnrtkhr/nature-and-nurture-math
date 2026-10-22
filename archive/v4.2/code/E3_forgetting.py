"""E-3 第8章 図8-1：破滅的忘却と、それを防ぐ学習（20個の重みの線形回帰）"""
import numpy as np
rng = np.random.default_rng(0); D = 20
wA = rng.normal(0, 1, D); wB = wA.copy()
wB[:6] = rng.normal(0, 1, 6); wB[12:] = rng.normal(0, 1, 8)   # 課題Bは一部の重みで違う答えを求める
sA = np.ones(D); sA[12:] = 0.05                                 # 課題Aは13〜20番の入力をほとんど使わない
XA = rng.normal(0, 1, (2000, D)) * sA; XB = rng.normal(0, 1, (2000, D))
yA, yB = XA @ wA, XB @ wB
lossA = lambda w: np.mean((XA[:500] @ w - yA[:500]) ** 2)
lossB = lambda w: np.mean((XB[:500] @ w - yB[:500]) ** 2)

def run(lam, steps=400, eta=0.01, seed=1):
    rg = np.random.default_rng(seed); w = np.zeros(D)
    for _ in range(steps):                                      # 課題Aを学ぶ
        i = rg.integers(0, 2000, 16); w -= eta * XA[i].T @ (XA[i] @ w - yA[i]) / 16
    wstar = w.copy(); F = np.mean(XA ** 2, axis=0)              # 重みごとの重要さ（入力の二乗の平均）
    for _ in range(steps):                                      # 課題Bを学ぶ（lam>0なら課題Aを守る）
        i = rg.integers(0, 2000, 16)
        w -= eta * (XB[i].T @ (XB[i] @ w - yB[i]) / 16 + lam * F * (w - wstar))
    return lossA(w), lossB(w)

for lam, name in [(0.0, "ふつうの学習"), (3.0, "守る学習")]:
    a, b = run(lam); print(f"{name}: 課題Aのロス {a:.2f}, 課題Bのロス {b:.2f}")
# 一つ変えてみる：wB[:6] = wA[:6] として、共有する重みで答えが食い違わないようにすると、守る代償は消えるか
