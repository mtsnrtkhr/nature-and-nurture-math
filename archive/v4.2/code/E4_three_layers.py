"""付録E-4（第4.2版）：三つの層と三つの現れ方（図10-1・10-2、第10章10.5の表、第11章11.4の表、応用1の表）

使い方：python E4_three_layers.py
必要なもの：Python 3、numpy（第11章の表には scipy、図を描くときは matplotlib。なければその部分を飛ばす）

模型（第10章）
  答えは多くの部分からなり、割合 f は固定成分（生まれつき正しい。誤り 0 から始まる）、
  残りは学ぶべき部分（初めの誤りの分散 1）。経験の雑音の分散 r = 1、一生 T = 50 回。
  誤りの分散は、更新の前に採点する（累積ロス）。

三つの現れ方（第4版）
  遅れ：学ぶべき部分の、初めの誤りがまだ残っている分     (1-η)^2 で減る項
  ぶれ：学ぶべき部分が、経験の雑音に合わせて揺れる分     η^2 r で積み上がる項
  損傷：固定成分（動かす必要のない部分）を、学習が動かした分
  この模型では、初めの誤りと雑音が独立なので、学ぶべき部分の誤りは
  「遅れ＋ぶれ」にちょうど分かれる（交差項の期待値は 0）。
"""
import numpy as np

T, r = 50, 1.0


def excess(v0, eta):                 # 一生の平均の余分な誤り（付録E の「コードの読み方の例」）
    v, s = v0, 0.0
    for _ in range(T):
        s += v                       # 更新の前に採点する（累積ロス）
        v = (1 - eta)**2 * v + eta**2 * r
    return s / T


def columns(etas, f):
    """学習率の列 etas（長さ T、全部の部分に共通）を使ったときの
    （遅れ, ぶれ, 損傷）を、一生の平均の余分な誤りとして返す。"""
    tr, nu, nc = 1.0, 0.0, 0.0   # 遅れの項, 学ぶべき部分の雑音, 固定成分の雑音
    D = F = G = 0.0
    for e in etas:
        D += tr; F += nu; G += nc          # 更新の前に採点する
        tr = (1 - e) ** 2 * tr
        nu = (1 - e) ** 2 * nu + e ** 2 * r
        nc = (1 - e) ** 2 * nc + e ** 2 * r
    return np.array([(1 - f) * D, (1 - f) * F, f * G]) / T


def ideal(f):
    """理想の学習者：固定成分は動かさず（学習率 0）、学ぶべき部分をカルマンの利得 1/(t+2) で学ぶ。"""
    d, fl, _ = columns([1 / (t + 2) for t in range(T)], 0.0)
    return np.array([(1 - f) * d, (1 - f) * fl, 0.0])


def best_constant(f, grid=np.linspace(0.001, 0.8, 8000)):
    tot = [columns(np.full(T, e), f).sum() for e in grid]
    e = grid[int(np.argmin(tot))]
    return e, columns(np.full(T, e), f)


def best_time_varying(f):
    """全部の部分に共通の学習率。ただし時間とともに変えてよい（第11章11.4の表）。"""
    from scipy.optimize import minimize
    e0, _ = best_constant(f)
    x0 = np.log(np.full(T, e0) / (1 - e0))
    obj = lambda x: columns(1 / (1 + np.exp(-x)), f).sum()
    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 5000})
    return columns(1 / (1 + np.exp(-res.x)), f)


def show_table(rows):
    names = ["遅れ", "ぶれ", "損傷"]
    print("            " + "".join(f"{n:>9}" for n in names) + "      行の和")
    for label, v in rows:
        print(f"{label:<8}" + "".join(f"{x:>10.3f}" for x in v) + f"{v.sum():>11.3f}")


def draw_figures(f, I, eta_b, B, A):
    """図10-1・10-2 を描いて保存する（matplotlib があるとき）。"""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 見つからないフォントの警告を出さない
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic", "Noto Sans JP", "sans-serif"]
    ink, ink2, line, bg = "#1f1f1f", "#5f5f5a", "#d9d9d4", "#fcfcfb"
    blue, orange, green = "#2a73d4", "#e8643a", "#17a673"
    badge = dict(boxstyle="round,pad=0.35", fc="#f3f3f0", ec="#555555", lw=0.9)

    def frame(ax):
        ax.set_facecolor(bg)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(line)
        ax.tick_params(colors=ink2, labelsize=10)
        ax.grid(axis="y", color="#e4e4e0", lw=0.8)
        ax.set_xlabel("全部の部分に共通の学習率 η", color=ink2, fontsize=11)
        ax.set_ylabel("一生の平均の余分な誤り", color=ink2, fontsize=11)
        ax.set_xlim(0, 0.6)

    grid = np.linspace(0.005, 0.6, 400)
    V = np.array([columns(np.full(T, e), f) for e in grid])
    tot = V.sum(1)
    mk = np.linspace(0, len(grid) - 1, 10).astype(int)

    # 図10-1：遅れ、ぶれ・損傷（この例では同じ大きさで、一本の線に重なる）、合計
    assert np.allclose(V[:, 1], V[:, 2])
    fig, ax = plt.subplots(figsize=(8.7, 4.8), dpi=150, facecolor=bg)
    frame(ax)
    ax.plot(grid, V[:, 0], color=blue, lw=2.2, ls=(0, (1.2, 1.6)), marker="s", markevery=list(mk),
            ms=7, mfc="white", mec=blue, mew=1.8)
    ax.plot(grid, V[:, 1], color=orange, lw=2.2, ls=(0, (6, 3)), marker="^", markevery=list(mk),
            ms=7, mfc="white", mec=orange, mew=1.8)
    ax.plot(grid, tot, color=ink, lw=2.6, marker="o", markevery=list(mk), ms=7, mfc="white", mec=ink, mew=1.8)
    ax.axhline(I.sum(), color=green, lw=2.2, ls=(0, (8, 2, 1.5, 2)))
    ax.plot([eta_b], [B.sum()], "o", color=ink, ms=5)
    ax.text(0.032, 0.39, "遅れ：知らない部分を\nなかなか覚えない", fontsize=11, color=ink, va="top")
    ax.text(0.423, 0.25, "ぶれと損傷：雑音で答えを動かす\n（この例では同じ大きさで、重なる）",
            fontsize=10.5, color=ink, va="top")
    ax.text(0.30, 0.245, "合計", fontsize=11, color=ink, fontweight="bold")
    ax.text(0.595, I.sum() + 0.008, "固定成分の位置を知り、部分ごとに最適に学ぶ学習者（理想）",
            fontsize=10.5, color=ink, ha="right", va="bottom")
    ax.set_ylim(0, 0.5)
    ax.text(0.995, 1.06, "説明用の数値例（計算による。実データではない）", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color=ink, bbox=badge)
    fig.tight_layout()
    fig.savefig("fig10_1_delay_damage.png", facecolor=bg)
    plt.close(fig)

    # 図10-2：図10-1 の合計を、三つの層に分ける
    fig, ax = plt.subplots(figsize=(8.7, 4.85), dpi=150, facecolor=bg)
    frame(ax)
    ax.plot(grid, tot, color=ink, lw=2.6)
    ax.axhline(B.sum(), color=blue, lw=2.2, ls=(0, (1.2, 1.6)))
    ax.axhline(I.sum(), color=green, lw=2.4, ls=(0, (6, 3)))
    ax.plot([eta_b], [B.sum()], "o", color=blue, ms=6, zorder=5)
    ax.plot([0.3], [A.sum()], "o", color=ink, ms=6, zorder=5)
    ax.text(0.285, A.sum() + 0.006, "実際の学習者（η = 0.3）", fontsize=11, color=ink, ha="right", va="bottom")
    x = 0.312
    for lo, hi, lab in ((B.sum(), A.sum(), f"取り逃がし {A.sum() - B.sum():.3f}"),
                        (I.sum(), B.sum(), f"届かない {B.sum() - I.sum():.3f}"),
                        (0.0, I.sum(), f"知らない {I.sum():.3f}")):
        ax.annotate("", (x, hi - 0.003), (x, lo + 0.003),
                    arrowprops=dict(arrowstyle="<->", color=orange, lw=1.8))
        ax.text(x + 0.012, (lo + hi) / 2, lab, fontsize=11.5, color=ink, fontweight="bold", va="center")
    ax.text(0.598, B.sum() + 0.004, "共通の学習率で最良", fontsize=10.5, color=ink, ha="right", va="bottom")
    ax.text(0.598, I.sum() + 0.004, "理想の学習者", fontsize=10.5, color=ink, ha="right", va="bottom")
    ax.text(0.598, 0.006, "神託", fontsize=10.5, color=ink2, ha="right", va="bottom")
    ax.text(0.026, 0.345, "左にずれると\n主に遅れが増える", fontsize=11, color=ink2, va="top")
    ax.text(0.33, 0.37, "右にずれると\n主にぶれと損傷が増える", fontsize=11, color=ink2, va="top")
    ax.set_ylim(0, 0.42)
    ax.text(0.995, 1.06, "説明用の数値例（計算による。実データではない）", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color=ink, bbox=badge)
    fig.tight_layout()
    fig.savefig("fig10_2_three_layers.png", facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    f = 0.5
    oracle = np.zeros(3)
    I = ideal(f)
    eta_b, B = best_constant(f)
    A = columns(np.full(T, 0.3), f)
    # 付録E の excess で書いた合計と、三つの列の和は一致する
    assert abs(f * excess(0, 0.3) + (1 - f) * excess(1, 0.3) - A.sum()) < 1e-12

    print(f"共通の学習率で最良：η = {eta_b:.3f}（余分な誤り {B.sum():.3f}）")
    print(f"理想の学習者 {I.sum():.3f}、実際の学習者（η = 0.3）{A.sum():.3f}\n")
    print("第10章10.5の表（三つの層 × 三つの現れ方）")
    show_table([("知らない", I - oracle), ("届かない", B - I), ("取り逃がし", A - B)])
    print("（四捨五入のため、欄の和が行の和と 0.001 ずれることがある）\n")

    # 証人（witness）による下限：Π の中の別の学び方 π' を一つ示せば、第3層の下限が言える
    for e_w in (0.15, 0.2):
        W = columns(np.full(T, e_w), f).sum()
        print(f"証人 η' = {e_w}：第3層 ≥ {A.sum() - W:.3f}（実際の第3層 {A.sum() - B.sum():.3f}）")

    # 慎重すぎる学習者（η = 0.05）：取り逃がしの行と、各列の値（第10章10.5、「機械学習の読者へ」）
    C = columns(np.full(T, 0.05), f)
    print("\n慎重すぎる学習者（η = 0.05）の取り逃がしの行")
    show_table([("取り逃がし", C - B)])
    print(f"各列の値：遅れ {C[0]:.3f}、ぶれ {C[1]:.3f}、損傷 {C[2]:.3f}、合計 {C.sum():.3f}"
          f"（最良の学習者は 遅れ {B[0]:.3f}、ぶれ {B[1]:.3f}、損傷 {B[2]:.3f}）")

    try:
        print("\n第11章11.4の表（共通の学習率、時間とともに変えてよい）")
        for ff in (0.0, 0.5, 0.9):
            Iff = ideal(ff); Bff = best_time_varying(ff)
            d = Bff - Iff
            print(f"f = {ff}: 第1層 {Iff.sum():.3f}  第2層 {d.sum():.3f}  "
                  f"第2層の内訳（遅れ {d[0]:+.3f}、ぶれ {d[1]:+.3f}、損傷 {d[2]:+.3f}）")
    except ImportError:
        print("（scipy がないので第11章の表は省略）")

    # 応用編・応用1：第10章の例を二つの目的（新しい課題＝遅れ+ぶれ、古い答え＝損傷）として読み、重みを変える
    print("\n応用1：重みを変えたときの、共通の学習率の最良（新しい課題の損失, 古い答えの損失）")
    I2 = np.array([I[0] + I[1], I[2]])
    print(f"理想の学習者            {I2[0]:.3f}  {I2[1]:.3f}")
    grid_w = np.linspace(0.001, 0.8, 8000)
    O = np.array([[v[0] + v[1], v[2]] for v in (columns(np.full(T, e), f) for e in grid_w)])
    for w_new, label in ((0.5, "1 対 1"), (0.8, "4 対 1"), (0.2, "1 対 4")):
        k = int(np.argmin(w_new * O[:, 0] + (1 - w_new) * O[:, 1]))
        print(f"重み {label}: η = {grid_w[k]:.3f}  {O[k,0]:.3f}  {O[k,1]:.3f}  "
              f"第2層の内訳（新 {O[k,0]-I2[0]:+.3f}、旧 {O[k,1]-I2[1]:+.3f}）")
    k_min = int(np.argmin(O[:, 0]))
    print(f"共通の学習率で届く、新しい課題の損失の最小 {O[k_min, 0]:.3f}（η = {grid_w[k_min]:.2f}）")
    # パレートフロント（η が 0 から k_min まで）が凸か：傾きが単調なら、どの点も重みつきの和で拾える
    s = np.diff(O[:k_min + 1, 1]) / np.diff(O[:k_min + 1, 0])
    print(f"フロントは凸か：{bool(np.all(np.diff(s) <= 1e-9))}")

    # 守るべき部分を見分けなければならない世界（第10章10.5「制約が、守ることに役立つとき」）
    # 各部分が確率 1/2 で固定成分（初めの誤り 0）か学ぶべき部分（初めの誤りの分散 1）になり、
    # 学習者はどちらかを知らない。神託は知っているので、余分な誤りは 0。
    rng_b = np.random.default_rng(1); Nb = 200_000
    fixed = rng_b.random(Nb) < 0.5
    e0 = np.where(fixed, 0.0, rng_b.normal(0, 1, Nb))
    noise = rng_b.normal(0, np.sqrt(r), (T, Nb))

    def world_b(answer):
        """answer(t, S, est) -> 次に採点する答え。S はそれまでの経験の和。(損傷, 動くべき部分の損失) を返す。"""
        S = np.zeros(Nb); est = np.zeros(Nb); out = np.zeros(2)
        for t in range(T):
            est = answer(t, S, est)
            err = (est - e0) ** 2
            out += [err[fixed].sum() / Nb, err[~fixed].sum() / Nb]
            S += e0 + noise[t]
        return out / T

    def bayes(t, S, est):            # 混合の事前分布の下での事後平均（理想の学習者）
        if t == 0:
            return np.zeros(Nb)
        yb = S / t; v = r / t
        l0 = np.exp(-yb ** 2 / (2 * v)) / np.sqrt(v)
        l1 = np.exp(-yb ** 2 / (2 * (1 + v))) / np.sqrt(1 + v)
        return l1 / (l0 + l1) * yb / (1 + v)

    def capped(cap):                 # 学習率が一定で、上限 cap の学習者（上限が効くので cap そのものが最良）
        def f(t, S, est):
            if t == 0:
                return np.zeros(Nb)
            y_prev = e0 + noise[t - 1]
            return est + cap * (y_prev - est)
        return f

    Ib = world_b(bayes); Cb = world_b(capped(0.02))
    print("\n守るべき部分を見分ける世界（損傷, 動くべき部分の損失）")
    print(f"理想の学習者           {Ib[0]:.3f}  {Ib[1]:.3f}  合計 {Ib.sum():.3f}")
    print(f"学習率の上限 0.02      {Cb[0]:.3f}  {Cb[1]:.3f}  合計 {Cb.sum():.3f}")
    print(f"「届かない」の行       損傷 {Cb[0]-Ib[0]:+.3f}、動くべき部分 {Cb[1]-Ib[1]:+.3f}、行の和 {Cb.sum()-Ib.sum():+.3f}")

    # モンテカルロ法で確かめる（1）：三つの列（η = 0.3）
    rng = np.random.default_rng(0); N = 200_000
    def mc(etas):
        a = rng.normal(0, 1, N); b = np.zeros(N); c = np.zeros(N); s = np.zeros(3)
        for e in etas:
            s += [np.mean(a ** 2), np.mean(b ** 2), np.mean(c ** 2)]
            a = (1 - e) * a
            b = (1 - e) * b + e * rng.normal(0, 1, N)
            c = (1 - e) * c + e * rng.normal(0, 1, N)
        return s / T * np.array([1 - f, 1 - f, f])
    M = mc(np.full(T, 0.3))
    print("\nモンテカルロ法（η = 0.3）：", np.round(M, 3), "合計", round(M.sum(), 3))

    # モンテカルロ法で確かめる（2）：4000世代、部分200個（半分は固定成分）で、理想・共通の最良・実際（図10-2）
    G, Np = 4000, 200
    u = rng.normal(0, 1, (G, Np)); u[:, Np // 2:] = 0.0      # 後半は固定成分（誤り 0 から出発）
    def mc_parts(rule):
        w = np.zeros((G, Np)); s = 0.0
        for t in range(T):
            s += np.mean((w - u) ** 2)
            y = u + rng.normal(0, np.sqrt(r), (G, Np))
            w += rule(t) * (y - w)
        return s / T
    kal = lambda t: np.concatenate([np.full(Np // 2, 1 / (t + 2)), np.zeros(Np // 2)])
    print(f"モンテカルロ法（4000世代、部分200個）：理想 {mc_parts(kal):.3f}、"
          f"共通の最良 {mc_parts(lambda t: eta_b):.3f}、実際 {mc_parts(lambda t: 0.3):.3f}")

    try:
        draw_figures(f, I, eta_b, B, A)
        print("\n図を fig10_1_delay_damage.png と fig10_2_three_layers.png に保存した")
    except ImportError:
        pass

# 一つ変えてみる：
#  1. f を 0.9 にすると、最良の共通の学習率と、表の各欄はどう変わるか（第11章11.4）
#  2. 実際の学習率を 0.05 にすると、取り逃がしの行のどの欄が負になるか（第10章10.5）
#  3. 証人 η' を 0.096 に近づけると、下限は第3層の値に近づくか
