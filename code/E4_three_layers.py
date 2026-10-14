"""E-4 第10章 図10-1・10-2、第11章の表：損失の三つの層（計算とモンテカルロ法）"""
import numpy as np
T, r, su = 50, 1.0, 1.0

def excess(v0, eta):                      # 一定の学習率で、初めの誤りの分散 v0 の部分を学ぶときの平均の誤り
    v, s = v0, 0.0
    for _ in range(T): s += v; v = (1 - eta) ** 2 * v + eta ** 2 * r
    return s / T

def layers(f):                            # f：生まれつき正しい（固定の）部分の割合
    ideal = (1 - f) * np.mean([1 / (1 / su + t / r) for t in range(T)])
    etas = np.linspace(0.001, 0.8, 4000)
    tot = np.array([f * excess(0, e) + (1 - f) * excess(su, e) for e in etas])
    return ideal, tot.min(), etas[tot.argmin()], f * excess(0, 0.3) + (1 - f) * excess(su, 0.3)

ideal, best, eta_b, actual = layers(0.5)
print(f"神託 0, 理想 {ideal:.3f}, 共通の学習率で最良 {best:.3f}（eta={eta_b:.3f}）, 実際（eta=0.3）{actual:.3f}")
print(f"三つの層：{ideal:.3f} / {best-ideal:.3f} / {actual-best:.3f}")

# モンテカルロ法：4000世代、部分200個（半分は固定成分）
rng = np.random.default_rng(0); G, N = 4000, 200
u = rng.normal(0, np.sqrt(su), (G, N)); u[:, N // 2:] = 0.0   # 後半は固定成分（誤り0から出発）
def mc(rule):
    w = np.zeros((G, N)); s = 0.0
    for t in range(T):
        s += np.mean((w - u) ** 2)
        y = u + rng.normal(0, np.sqrt(r), (G, N))
        w += rule(t) * (y - w)
    return s / T
kal = lambda t: np.concatenate([np.full(N // 2, 1 / (t + 2)), np.zeros(N // 2)])
print(f"モンテカルロ：理想 {mc(kal):.4f}, 共通の最良 {mc(lambda t: eta_b):.4f}, 実際 {mc(lambda t: 0.3):.4f}")
# 一つ変えてみる：layers(f) の f を 0, 0.5, 0.9 にして、第11章の表の第1層を確かめる
# （第2層の「時間とともに変えてよい共通の学習率」の値段は E-6 の方法で求める）
