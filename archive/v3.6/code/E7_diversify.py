"""E-7 発展6：二つの選択肢に分けると、なぜ増殖率が上がるか"""
import numpy as np
rng = np.random.default_rng(1); mu, sd, N = 0.05, 0.2, 400000
z = rng.standard_normal((N, 2))
for rho in (-0.8, 0.0, 0.8, 1.0):
    L = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]) + 1e-12 * np.eye(2))
    r = mu + sd * (z @ L.T)
    exact = lambda w: np.mean(np.log1p(r @ np.array([w, 1 - w])))
    approx = lambda w: mu - 0.5 * sd**2 * (w**2 + (1-w)**2 + 2*rho*w*(1-w))
    print(f"相関 {rho:+.1f}：全部賭け {exact(1):.4f}（近似 {approx(1):.4f}）、半分ずつ {exact(.5):.4f}（近似 {approx(.5):.4f}）")
# 一つ変えてみる：sd を 0.5 に上げ、近似が合わなくなることを確かめる
