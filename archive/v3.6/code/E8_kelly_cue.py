"""E-8 第6章、付録B.5：手がかりの価値は相互情報量（系統の増殖をシミュレーションで確かめる）"""
import numpy as np
rng = np.random.default_rng(2); G = 200000
P = np.array([0.7, 0.3]); f = np.array([2.0, 3.0])     # 環境の確率と、合った表現型の倍率
e = rng.choice(2, G, p=P)
def growth(pi_given_cue, cue):
    pi = pi_given_cue[cue, e]                          # 合った表現型に割り当てた割合
    return np.mean(np.log(pi * f[e]))
H = -np.sum(P * np.log(P))
no_cue = growth(np.tile(P, (2, 1)), np.zeros(G, int))          # 確率どおりに分ける
for acc in (1.0, 0.9, 0.6):                                    # 手がかりの正しさ
    cue = np.where(rng.random(G) < acc, e, 1 - e)
    joint = np.array([[P[0]*acc, P[1]*(1-acc)], [P[0]*(1-acc), P[1]*acc]])   # [cue, e]
    post = joint / joint.sum(1, keepdims=True)                              # P(e | cue)
    pc = joint.sum(1); I = np.sum(joint * np.log(joint / (pc[:, None] * P[None, :]) + 1e-300))
    print(f"手がかりの正しさ {acc}：増殖率の上がり幅 {growth(post, cue)-no_cue:.4f}、相互情報量 {I:.4f}")
print(f"手がかりなしの増殖率 {no_cue:.4f}（理論値 {np.sum(P*np.log(f))-H:.4f}）")
