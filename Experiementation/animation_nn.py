#!/usr/bin/env python3
"""
Sin vs ReLU — NN approximation of arccos(x)
• Drag NEURONS slider
• Click colour swatches to repaint each network's line
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, RadioButtons, Button

# ── palette ───────────────────────────────────────────────────────────────────
BG    = '#FFFFFF'
PANEL = '#F6F8FA'
BORD  = '#D0D7DE'
TEXT  = '#24292F'
GRID  = '#EAECEF'
C_TGT = '#111111'

# default network colours (user can change these)
state = {
    'sin_col': '#FF4081',
    'cmp_col': '#29B6F6',
}

# swatches available to pick from
SWATCHES = [
    '#FF4081', '#FF6B35', '#FFD700', '#39FF14',
    '#29B6F6', '#7C3AED', '#FF00FF', '#FFFFFF',
    '#00E5FF', '#76FF03', '#FF1744', '#F8BBD9',
]

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor':  PANEL,
    'axes.edgecolor':   BORD,  'axes.labelcolor': TEXT,
    'xtick.color':      TEXT,  'ytick.color':     TEXT,
    'text.color':       TEXT,  'grid.color':      GRID,
    'grid.linewidth':   0.5,   'axes.grid':       True,
    'axes.spines.top':  False, 'axes.spines.right': False,
    'font.size': 10,
})

# ── targets & activations ─────────────────────────────────────────────────────
TARGETS = {
    'arccos(x)':   (lambda x: np.arccos(x),                               (-0.99, 0.99)),
    'arcsin(x)':   (lambda x: np.arcsin(x),                               (-0.99, 0.99)),
    'sin(3x)':     (lambda x: np.sin(3 * x),                              (-np.pi, np.pi)),
    'Inverse MZM': (lambda x: (2/np.pi)*np.arcsin(np.clip(x,-0.99,0.99)),  (-0.99, 0.99)),
}
CMPS = {
    'ReLU':       lambda x: np.maximum(0, x),
    'Leaky ReLU': lambda x: np.where(x >= 0, x, 0.1 * x),
    'Tanh':       lambda x: np.tanh(x),
}

def sin_fn(x): return np.sin(x)

def fit(act, x, y, n, seed=42):
    rng = np.random.default_rng(seed)
    a   = rng.uniform(-6, 6, n)
    b   = rng.uniform(-np.pi*3, np.pi*3, n)
    Phi = act(np.outer(x, a) + b)
    Phi = np.hstack([Phi, np.ones((len(x), 1))])
    w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return Phi @ w

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8.2))
fig.text(0.5, 0.992,
    'How does a neural network approximate a smooth curve?  —  Sin vs ReLU',
    ha='center', fontsize=13, fontweight='bold', color=TEXT, va='top')

# two main axes: overlay fit + MSE curve
ax_fit = fig.add_axes([0.05,  0.27, 0.60, 0.63])
ax_mse = fig.add_axes([0.69,  0.27, 0.28, 0.63])

# slider
ax_sl  = fig.add_axes([0.27, 0.175, 0.46, 0.025])
slider = Slider(ax_sl, 'Neurons  ', 1, 80, valinit=4, valstep=1, color=state['sin_col'])
slider.label.set_color(TEXT);       slider.label.set_fontsize(11)
slider.valtext.set_color(TEXT);     slider.valtext.set_fontsize(12)
slider.valtext.set_fontweight('bold')

# radio: target
ax_rtgt = fig.add_axes([0.05, 0.02, 0.16, 0.13]); ax_rtgt.axis('off')
radio_tgt = RadioButtons(ax_rtgt, list(TARGETS.keys()), active=0, activecolor=state['sin_col'])
for lb in radio_tgt.labels: lb.set_fontsize(8.5); lb.set_color(TEXT)
ax_rtgt.set_title('Target function', color=TEXT, fontsize=8.5, pad=1)

# radio: compare
ax_ract = fig.add_axes([0.79, 0.02, 0.14, 0.10]); ax_ract.axis('off')
radio_cmp = RadioButtons(ax_ract, list(CMPS.keys()), active=0, activecolor=state['cmp_col'])
for lb in radio_cmp.labels: lb.set_fontsize(8.5); lb.set_color(TEXT)
ax_ract.set_title('Compare against', color=TEXT, fontsize=8.5, pad=1)

# ── colour swatch rows ────────────────────────────────────────────────────────
# label text
fig.text(0.235, 0.125, 'Sin colour →',   color=TEXT, fontsize=8.5, ha='right', va='center')
fig.text(0.235, 0.075, 'Cmp colour →',   color=TEXT, fontsize=8.5, ha='right', va='center')

swatch_btns_sin = []
swatch_btns_cmp = []

for i, col in enumerate(SWATCHES):
    x0 = 0.245 + i * 0.038
    # sin row
    ax_s = fig.add_axes([x0, 0.108, 0.030, 0.030])
    b_s  = Button(ax_s, '', color=col, hovercolor=col)
    # outline the active one
    for spine in ax_s.spines.values():
        spine.set_edgecolor(TEXT if col == state['sin_col'] else col)
        spine.set_linewidth(2.5 if col == state['sin_col'] else 0.8)
    swatch_btns_sin.append((b_s, ax_s, col))

    # cmp row
    ax_c = fig.add_axes([x0, 0.058, 0.030, 0.030])
    b_c  = Button(ax_c, '', color=col, hovercolor=col)
    for spine in ax_c.spines.values():
        spine.set_edgecolor(TEXT if col == state['cmp_col'] else col)
        spine.set_linewidth(2.5 if col == state['cmp_col'] else 0.8)
    swatch_btns_cmp.append((b_c, ax_c, col))

def make_sin_cb(chosen_col):
    def cb(_):
        state['sin_col'] = chosen_col
        # update outlines
        for _, ax_sw, c in swatch_btns_sin:
            for sp in ax_sw.spines.values():
                sp.set_edgecolor(TEXT if c == chosen_col else c)
                sp.set_linewidth(2.5 if c == chosen_col else 0.8)
        redraw()
    return cb

def make_cmp_cb(chosen_col):
    def cb(_):
        state['cmp_col'] = chosen_col
        for _, ax_sw, c in swatch_btns_cmp:
            for sp in ax_sw.spines.values():
                sp.set_edgecolor(TEXT if c == chosen_col else c)
                sp.set_linewidth(2.5 if c == chosen_col else 0.8)
        redraw()
    return cb

for b_s, _, col in swatch_btns_sin:
    b_s.on_clicked(make_sin_cb(col))
for b_c, _, col in swatch_btns_cmp:
    b_c.on_clicked(make_cmp_cb(col))

# ── MSE cache ─────────────────────────────────────────────────────────────────
_mse_cache = {}

# ── main draw ─────────────────────────────────────────────────────────────────
def redraw(_=None):
    n        = int(slider.val)
    tgt_name = radio_tgt.value_selected
    cmp_name = radio_cmp.value_selected
    tgt_fn, (lo, hi) = TARGETS[tgt_name]
    cmp_act           = CMPS[cmp_name]
    cs = state['sin_col']
    cc = state['cmp_col']

    # high-res x for smooth target + visible kinks
    X  = np.linspace(lo, hi, 600)
    Yt = tgt_fn(X)
    Ys = fit(sin_fn,  X, Yt, n)
    Yc = fit(cmp_act, X, Yt, n)

    mse_s = float(np.mean((Ys - Yt)**2))
    mse_c = float(np.mean((Yc - Yt)**2))

    yall = np.concatenate([Yt, Ys, Yc])
    pad  = (yall.max() - yall.min()) * 0.14
    ylim = (yall.min() - pad, yall.max() + pad)

    # ── overlay plot: target + both approximations ────────────────────────────
    ax_fit.cla()

    # soft error fills (kept subtle so the overlay stays readable)
    ax_fit.fill_between(X, Yt, Ys, alpha=0.14, color=cs, interpolate=True, zorder=1)
    ax_fit.fill_between(X, Yt, Yc, alpha=0.06, color=cc, interpolate=True, zorder=1)

    # target — thin dashed white
    ax_fit.plot(X, Yt, color=C_TGT, lw=1.3, ls='--', alpha=0.80, label='Target', zorder=4)

    # overlays — distinct line styles so colour-blind and print-friendly
    # Sin: add a faint "glow" so it reads as the hero curve
    ax_fit.plot(X, Ys, color=cs, lw=6.0, alpha=0.12, zorder=5)
    ax_fit.plot(X, Ys, color=cs, lw=2.6,
                label=f'Sin NN  (MSE={mse_s:.4g})', zorder=7)

    # Compare: outline + sharp joins (makes ReLU kinks pop)
    ax_fit.plot(X, Yc, color=PANEL, lw=4.0, alpha=0.95,
                solid_joinstyle='miter', solid_capstyle='butt',
                antialiased=False, zorder=5.5)
    ax_fit.plot(X, Yc, color=cc, lw=2.1, ls='-.',
                solid_joinstyle='miter', solid_capstyle='butt',
                antialiased=False,
                label=f'{cmp_name} NN  (MSE={mse_c:.4g})', zorder=6)

    ax_fit.set_ylim(ylim)
    ax_fit.set_xlabel('x', fontsize=10)
    ax_fit.set_ylabel(tgt_name, fontsize=10)
    ax_fit.set_title(f'Overlay fit  ·  {n} neuron{"s" if n>1 else ""}  (Sin vs {cmp_name})',
                     color=TEXT, fontweight='bold', fontsize=12, pad=4)
    ax_fit.legend(fontsize=10, loc='best', framealpha=0.20)

    # winner ribbon
    ratio = mse_c / max(mse_s, 1e-12)
    # if ratio > 1.05:
    #     _ribbon(ax_fit, f'Sin: ↓ {ratio:.1f}× lower error', cs)
    # elif ratio < 0.95:
    #     _ribbon(ax_fit, f'{cmp_name}: ↓ {1/ratio:.1f}× lower error', cc)

    # ── MSE curve ─────────────────────────────────────────────────────────────
    cache_key = (tgt_name, cmp_name)
    if cache_key not in _mse_cache:
        ns   = list(range(1, 81, 2))
        ms_s = [np.mean((fit(sin_fn,  X, Yt, nn) - Yt)**2) for nn in ns]
        ms_c = [np.mean((fit(cmp_act, X, Yt, nn) - Yt)**2) for nn in ns]
        _mse_cache[cache_key] = (ns, ms_s, ms_c)

    ns, ms_s_all, ms_c_all = _mse_cache[cache_key]

    ax_mse.cla()
    ax_mse.semilogy(ns, ms_c_all, color=cc, lw=1.8, label=cmp_name, zorder=3)
    ax_mse.semilogy(ns, ms_s_all, color=cs, lw=1.8, label='Sin',    zorder=4)
    # ax_mse.fill_between(ns, ms_s_all, ms_c_all,
    #                     where=[c >= s for s, c in zip(ms_s_all, ms_c_all)],
    #                     alpha=0.18, color=cs, interpolate=True,
    #                     label='Sin advantage')

    cur_s = float(np.mean((fit(sin_fn,  X, Yt, n) - Yt)**2))
    cur_c = float(np.mean((fit(cmp_act, X, Yt, n) - Yt)**2))
    ax_mse.axvline(n, color=TEXT, lw=1.0, ls='--', alpha=0.4, zorder=2)
    ax_mse.scatter([n], [cur_s], color=cs, s=70, zorder=6)
    ax_mse.scatter([n], [cur_c], color=cc, s=70, zorder=6)
    ax_mse.annotate(f'n={n}', xy=(n, min(cur_s, cur_c)),
                    xytext=(n + 2.5, min(cur_s, cur_c)),
                    fontsize=8.5, color=TEXT, va='center')

    ax_mse.set_xlabel('Number of neurons', fontsize=10)
    ax_mse.set_ylabel('MSE  (log scale)',  fontsize=10)
    ax_mse.set_title('Error as neuron count grows',
                     color=TEXT, fontweight='bold', fontsize=12, pad=8)
    ax_mse.legend(fontsize=9.5)
    ax_mse.set_xlim(0, 83)

    fig.canvas.draw_idle()


def _ribbon(ax, msg, color):
    ax.text(0.97, 0.95, msg,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.4', fc=color+'1A', ec=color, lw=2.0))


# ── wire ──────────────────────────────────────────────────────────────────────
slider.on_changed(redraw)
radio_tgt.on_clicked(lambda _: (_mse_cache.clear(), redraw()))
radio_cmp.on_clicked(lambda _: (_mse_cache.clear(), redraw()))

redraw()
plt.show()