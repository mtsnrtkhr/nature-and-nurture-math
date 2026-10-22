"""E-5 発展1：軸がそろうとき、ずれるとき（行列のカルマンフィルタと、二つの課題）"""
import numpy as np
rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def posterior(P0, R, t):                       # 答えをじかに観測する（観測 = 答え + 雑音）、Q = 0
    P = P0.copy()
    for _ in range(t):
        K = P @ np.linalg.inv(P + R); P = (np.eye(2) - K) @ P
    return P

P0 = np.diag([4.0, 0.05])
print("そろう：", np.round(posterior(P0, np.eye(2), 10), 5).tolist(),
      " 1変数の式：", [round(1 / (1 / p + 10), 5) for p in (4.0, 0.05)])
Rrot = rot(0.6) @ np.diag([0.2, 5.0]) @ rot(0.6).T
print("ずれる：", np.round(posterior(P0, Rrot, 10), 4).tolist())

# 二つの課題：A は a·w = 1、B は b·w = -1、a と b のなす角 theta
for deg in (15, 30, 45, 60, 90):
    th = np.radians(deg)
    plain = (np.cos(th) * (1 + np.cos(th))) ** 2       # B をふつうに学んだ後の A のロス
    step = (1 + np.cos(th)) / np.sin(th)               # A を守って B を学ぶのに要る動き
    print(f"{deg:3d}度：ふつうの学習で A のロス {plain:.3f}、守る学習の動き {step:.3f}")
# 一つ変えてみる：rot(0.6) の角度を 0 から 1.57 まで変え、非対角の成分の大きさを見る
