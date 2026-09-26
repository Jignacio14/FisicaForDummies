"""Cálculos del TP1 (densidad y péndulo) y generación de las figuras de img/.

Criterio de incertezas (igual en todo el informe):
  * Magnitudes repetidas: DX = (X_max + min - (X_min - min)) / 2, con min la
    apreciación de la magnitud; si la incerteza instrumental no estadística
    (p. ej. el tiempo de reacción) es mayor, se adopta esa.
  * Magnitudes indirectas: propagación lineal con derivadas parciales.
"""
import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
AZUL, NARANJA, GRIS = "#2a76d2", "#e8663a", "#999999"

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#777777", "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "axes.grid": True, "grid.color": "#e6e6e6",
    "axes.formatter.use_locale": False,
})


def coma(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def dispersion(x, apreciacion):
    return (x.max() - x.min()) / 2 + apreciacion


# ---------------------------------------------------------------- densidad
V = np.array([48 - 30, 56 - 40, 57 - 40, 45 - 29], float)   # ml
m = np.array([45.70, 45.70, 45.71, 45.71])                    # g
dm = max(dispersion(m, 0.01), 0.01)
dV = max(dispersion(V, 4.0), 4.0)          # apreciación de V: 2 + 2 ml
Vp, mp = V.mean(), m.mean()
dens = mp / Vp
ddens = dm / Vp + mp / Vp**2 * dV
print(f"m = {mp:.3f} ± {dm:.3f} g   V = {Vp:.2f} ± {dV:.1f} ml")
print(f"densidad = {dens:.3f} ± {ddens:.3f} g/cm3 ({ddens / dens:.1%})")

# ------------------------------------------------------------------ péndulo
L = np.array([96.5, 88, 78, 58.5, 43, 87]) / 100               # m
dL = 0.001
t = np.array([[20.10, 19.60, 20.10], [19.77, 18.91, 18.72], [18.20, 17.96, 17.79],
              [15.71, 15.43, 15.48], [13.92, 14.12, 14.18], [18.21, 18.50, 18.45]])
N = 10
dt = np.array([max(dispersion(ti, 0.01), 0.21) for ti in t])
T, dT = t.mean(axis=1) / N, dt / N
T2, dT2 = T**2, 2 * T * dT
g = 4 * PI**2 * L / T2
eg = dL / L + 2 * dT / T
dg = g * eg
for i in range(len(L)):
    print(f"L={L[i]*100:5.1f} dt={dt[i]:.3f} T={T[i]:.4f} T2={T2[i]:.3f}±{dT2[i]:.3f} "
          f"g={g[i]:.3f}±{dg[i]:.3f} ({eg[i]:.2%})")


def ajuste(x, y):
    n = len(x)
    a, b = np.polyfit(x, y, 1)
    r = y - (a * x + b)
    s2 = r @ r / (n - 2)
    sxx = ((x - x.mean())**2).sum()
    da = np.sqrt(s2 / sxx)
    db = np.sqrt(s2 * (1 / n + x.mean()**2 / sxx))
    r2 = 1 - r @ r / ((y - y.mean())**2).sum()
    return a, da, b, db, r2


usados = np.array([1, 1, 1, 1, 0, 0], bool)
fits = {"todos": ajuste(L, T2), "sin": ajuste(L[usados], T2[usados])}
for k, (a, da, b, db, r2) in fits.items():
    G, dG = 4 * PI**2 / a, 4 * PI**2 / a**2 * da
    print(f"{k}: a={a:.4f}±{da:.4f} b={b:.4f}±{db:.4f} R2={r2:.5f} g={G:.3f}±{dG:.3f} ({dG/G:.2%})")

# Incerteza de la pendiente propagando las barras de T^2 (ajuste pesado)
w = 1 / dT2[usados]**2
xw = (w * L[usados]).sum() / w.sum()
da_barras = 1 / np.sqrt((w * (L[usados] - xw)**2).sum())
a_sin = fits["sin"][0]
print(f"Da por barras = {da_barras:.3f} -> Dg = {4*PI**2/a_sin**2*da_barras:.2f}")

# ------------------------------------------------------------------ figuras
fig, ax = plt.subplots(figsize=(6.4, 2.4), dpi=200)
filas = ["Medición (probeta)", "Aluminio", "Latón", "Bronce"]
ax.axvspan(dens - ddens, dens + ddens, color=AZUL, alpha=0.09, lw=0)
ax.errorbar(dens, 0, xerr=ddens, fmt="o", color=AZUL, capsize=3, ms=7, lw=1.5)
for y, (c, e) in enumerate([(2.70, 0.02), (8.55, 0.15), (8.80, 0.10)], start=1):
    ax.errorbar(c, y, xerr=e, fmt="o", color=GRIS, mfc="white", mew=1.8, capsize=3, lw=1.5)
ax.set_yticks(range(4), filas)
ax.set_ylim(3.6, -0.6)
ax.set_xlim(1, 10)
ax.grid(axis="y", visible=False)
ax.set_xlabel("δ [g/cm³]")
fig.tight_layout()
fig.savefig("img/fig1-densidad.png")

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
G, dG = 4 * PI**2 / a_sin, 4 * PI**2 / a_sin**2 * fits["sin"][1]
etiquetas = [f"L = {coma(l*100, 1).removesuffix(',0')} cm" for l in L] + ["Regresión lineal"]
valores = list(zip(g, dg)) + [(G, dG)]
for y, (v, e) in enumerate(valores):
    descartado = y < len(L) and not usados[y]
    color = NARANJA if descartado else AZUL
    ax.errorbar(v, y, xerr=e, fmt="o", color=color, mfc="white" if descartado else color,
                mew=1.8, capsize=4, ms=7, lw=1.5)
    nd = 1 if e >= 0.3 else 2          # 1 cifra significativa salvo si empieza en 1 o 2
    ax.text(v + e + 0.05, y, f"{coma(v, nd)} ± {coma(e, nd)}", va="center", fontsize=9,
            color="#333333")
ax.axvline(9.80, color="#888888", ls="--", lw=1.2)
ax.text(9.80, -1.05, "g de referencia\n(Buenos Aires) ≈ 9,80", ha="center", va="center", fontsize=9,
        color="#555555")
ax.set_yticks(range(len(etiquetas)), etiquetas)
ax.set_ylim(len(etiquetas) - 0.3, -1.9)
ax.set_xlim(8.1, 10.9)
ax.grid(axis="y", visible=False)
ax.set_xlabel("g [m/s²]")
fig.tight_layout()
fig.savefig("img/fig2-gravedad.png")

fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=200)
xs = np.linspace(0.38, 1.02, 2)
a6, _, b6, _, _ = fits["todos"]
a4, _, b4, _, _ = fits["sin"]
ax.plot(xs, a6 * xs + b6, "--", color=GRIS, lw=1.5,
        label=f"Ajuste con todos los puntos: T² = {coma(a6, 2)}·L + {coma(b6, 2)}")
ax.plot(xs, a4 * xs + b4, color=AZUL, lw=2,
        label=f"Ajuste sin puntos descartados: T² = {coma(a4, 2)}·L + {coma(b4, 3)}")
ax.errorbar(L[usados], T2[usados], yerr=dT2[usados], fmt="o", color=AZUL, mec="white",
            capsize=3, ms=6, label="Mediciones utilizadas")
ax.errorbar(L[~usados], T2[~usados], yerr=dT2[~usados], fmt="o", color=NARANJA, mfc="white",
            mew=1.8, capsize=3, ms=7, label="Mediciones descartadas (L = 43 cm y 87 cm)")
ax.set_xlabel("L [m]")
ax.set_ylabel("T² [s²]")
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("img/fig3-regresion.png")
