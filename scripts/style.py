"""Shared typography, Nature-grade colour semantics, and export QA helpers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.patches import Polygon

# Register Arial if installed outside default font cache.
if 'Arial' not in {f.name for f in font_manager.fontManager.ttflist}:
    for _fp in font_manager.findSystemFonts():
        if Path(_fp).name.lower().startswith('arial'):
            font_manager.fontManager.addfont(_fp)

V = Path(__file__).resolve().parents[1]
SD = V / 'source_data'
FIG = V / 'figures'
QA = V / 'qa'
FIG.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

MM = 1.0 / 25.4
INK = '#1E293B'
MUTED = '#52606D'
GREY = '#AEB8C2'
GRID = '#E6EAEF'

# Semantic pathway & system colours (colourblind-aware, high grayscale contrast)
C = {
    'power': '#C4473A',     # Site power (warm terracotta red)
    'inject': '#1F9E89',    # Gas grid injection (teal)
    'truck': '#3B7EA1',     # CNG / LNG heavy-duty transport (steel blue)
    'reserve': '#7B68B0',   # Flexible reserve (muted violet)
    'switch': '#D49A3A',    # Switch-at-net-zero pathway (ochre gold)
    'red': '#C4473A',
    'teal': '#1F9E89',
    'blue': '#3B7EA1',
    'navy': '#24476B',
    'purple': '#7B68B0',
    'green': '#70A35E',
    'sand': '#D49A3A',
    'brown': '#5E3A2D',
    'grey': '#94A3B8',
    'light': '#E8EDF2',
}

# Target-year ramp (dark -> bright or deep navy -> teal)
YC = {2028: '#24476B', 2030: '#3B7EA1', 2033: '#58A6A6', 2035: '#1F9E89'}
YC_WARM = {2028: '#7F2E24', 2030: '#BA4C39', 2033: '#DE8548', 2035: '#E8B368'}

SEQ = colors.LinearSegmentedColormap.from_list('teal15', ['#F4F8F7', '#C0E1DA', '#5EB8A6', '#1F9E89', '#125C55'])
BLUE = colors.LinearSegmentedColormap.from_list('blue15', ['#F4F7FA', '#C7DAEA', '#7BA7CA', '#3B7EA1', '#1D3E5E'])
WARM = colors.LinearSegmentedColormap.from_list('warm15', ['#FBF6EE', '#F1D4A2', '#E59B56', '#C95E38', '#863022'])
DIV = colors.LinearSegmentedColormap.from_list('div15', ['#B34436', '#E59782', '#F8F6F2', '#82C8BA', '#1D7D6E'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7.2,
    'axes.labelsize': 7.2,
    'axes.titlesize': 7.4,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.55,
    'lines.linewidth': 1.15,
    'lines.markersize': 3.2,
    'patch.linewidth': 0.4,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.3,
    'ytick.major.size': 2.3,
    'xtick.major.pad': 2.0,
    'ytick.major.pad': 2.0,
    'axes.labelpad': 2.6,
    'legend.frameon': False,
    'text.color': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'axes.edgecolor': INK,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'mathtext.default': 'regular',
    'savefig.dpi': 600,
    'figure.dpi': 180,
})

SOURCES = {}
PANELS = {}
MODELED = {'AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'FR', 'DE', 'GR', 'IT', 'PT', 'RO', 'SI', 'ES', 'UK'}


def read(name: str) -> pd.DataFrame:
    p = SD / f'{name}.csv'
    SOURCES[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)


def read_json(name: str):
    p = SD / name if '.' in name else (SD / f'{name}.geojson' if (SD / f'{name}.geojson').exists() else SD / f'{name}.json')
    SOURCES[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def fig(h=185):
    return plt.figure(figsize=(180 * MM, h * MM))


def ax(f, x, y, w, h, **kw):
    fw, fh = f.get_size_inches() / MM
    return f.add_axes([x / fw, 1.0 - (y + h) / fh, w / fw, h / fh], **kw)


def label(f, a, s, dx_mm=0.0, dy_mm=1.6, x_abs=None, y_abs=None):
    f.canvas.draw()
    r = f.canvas.get_renderer()
    fw, fh = f.get_size_inches() / MM
    b = a.get_tightbbox(r).transformed(f.transFigure.inverted())
    x0 = (x_abs / fw) if x_abs is not None else max(0.004, b.x0 - dx_mm / fw)
    y0 = (1.0 - y_abs / fh) if y_abs is not None else (a.get_position().y1 + dy_mm / fh)
    f.text(x0, y0, s, weight='bold', size=9.0, va='bottom', ha='left')
    PANELS[s] = {'xlabel': a.get_xlabel(), 'ylabel': a.get_ylabel()}


def grid(a, axis='y'):
    a.grid(axis=axis, color=GRID, lw=0.45, zorder=0)
    a.set_axisbelow(True)


def yearaxis(a, short=False):
    a.set_xticks([2028, 2030, 2033, 2035])
    if short:
        a.set_xticklabels(["'28", "'30", "'33", "'35"])
    a.set_xlabel('Target year')


def band(a, d, y, col, label_txt=None, ls='-', marker='o', ms=3.0, alpha=0.14):
    q = d.groupby('year')[y].agg(['mean', 'min', 'max'])
    x = q.index.to_numpy()
    a.fill_between(x, q['min'], q['max'], color=col, alpha=alpha, lw=0, zorder=1)
    a.plot(x, q['mean'], color=col, ls=ls, marker=marker, ms=ms, label=label_txt, zorder=3)
    return q


def textheat(a, z, fmt='.2f', threshold=None, fs=7.0, invert=False):
    z = np.asarray(z, float)
    if threshold is None:
        threshold = (np.nanmax(z) + np.nanmin(z)) / 2.0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            if np.isfinite(v):
                txt = format(0.0 if abs(v) < 1e-9 else v, fmt)
                dark = (v < threshold) if invert else (v > threshold)
                a.text(j, i, txt, ha='center', va='center', fontsize=fs,
                       color='white' if dark else INK)


def repel(ys, gap):
    o = np.argsort(ys)
    out = np.asarray(ys, float).copy()
    for _ in range(300):
        moved = False
        for i0, i1 in zip(o[:-1], o[1:]):
            if out[i1] - out[i0] < gap:
                d = (gap - (out[i1] - out[i0])) / 2.0
                out[i0] -= d
                out[i1] += d
                moved = True
        if not moved:
            break
    return out


def ribbon(ax_obj, y, xs, ws, color, H=0.72, alpha=0.92, z=2):
    top = [(x, y + w * H / 2.0) for x, w in zip(xs, ws)]
    bot = [(x, y - w * H / 2.0) for x, w in zip(xs[::-1], ws[::-1])]
    ax_obj.add_patch(Polygon(top + bot, closed=True, facecolor=color, edgecolor='none', alpha=alpha, zorder=z))


def gradient_fill(ax_obj, x, y, color, y0=0.0, alpha_top=0.45, alpha_bottom=0.03, zorder=1):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xl, yl = ax_obj.get_xlim(), ax_obj.get_ylim()
    z = np.zeros((128, 1, 4))
    z[:, :, :3] = colors.to_rgb(color)
    z[:, 0, 3] = np.linspace(alpha_bottom, alpha_top, 128)
    lo = min(y0, np.nanmin(y))
    hi = np.nanmax(y)
    im = ax_obj.imshow(z, aspect='auto', extent=[x.min(), x.max(), lo, hi if hi > lo else lo + 1e-9],
                       origin='lower', zorder=zorder, interpolation='bilinear')
    poly = Polygon(np.r_[np.c_[x, y], [[x.max(), y0], [x.min(), y0]]], facecolor='none', edgecolor='none')
    ax_obj.add_patch(poly)
    im.set_clip_path(poly)
    ax_obj.set_xlim(xl)
    ax_obj.set_ylim(yl)
    return im


def laea(lon, lat):
    lo, la = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    l0, p0 = np.deg2rad(10.0), np.deg2rad(52.0)
    k = np.sqrt(2.0 / (1.0 + np.sin(p0) * np.sin(la) + np.cos(p0) * np.cos(la) * np.cos(lo - l0)))
    return (
        6371.0 * k * np.cos(la) * np.sin(lo - l0),
        6371.0 * k * (np.cos(p0) * np.sin(la) - np.sin(p0) * np.cos(la) * np.cos(lo - l0)),
    )


def _clip_rect(pts, x0, x1, y0, y1):
    def cut(P, inside, inter):
        out = []
        for i in range(len(P)):
            a, b = P[i - 1], P[i]
            if inside(b):
                if not inside(a):
                    out.append(inter(a, b))
                out.append(b)
            elif inside(a):
                out.append(inter(a, b))
        return out

    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    P = [tuple(q) for q in pts]
    for inside, inter in [
        (lambda q: q[0] >= x0, lambda a, b: ix(a, b, x0)),
        (lambda q: q[0] <= x1, lambda a, b: ix(a, b, x1)),
        (lambda q: q[1] >= y0, lambda a, b: iy(a, b, y0)),
        (lambda q: q[1] <= y1, lambda a, b: iy(a, b, y1)),
    ]:
        if not P:
            break
        P = cut(P, inside, inter)
    return np.asarray(P)


def countrymap(a, values, cmap=BLUE, norm=None, vmin=0.0, vmax=0.5, zero_color='#EEF2F6', frame=True):
    geo = read_json('countries')
    if norm is None:
        norm = colors.Normalize(vmin, vmax)
    bx0, by0 = laea(-10.5, 36.0)
    bx1, _ = laea(28.0, 36.0)
    _, by1 = laea(10.0, 60.5)
    X0, X1, Y0, Y1 = bx0 - 140.0, bx1 + 140.0, by0 - 70.0, by1 + 40.0

    for feature in geo['features']:
        k = feature['properties']['CNTR_ID']
        k = {'EL': 'GR', 'GB': 'UK'}.get(k, k)
        g = feature['geometry']
        polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates'] if g['type'] == 'MultiPolygon' else []
        for poly in polys:
            ring = np.asarray(poly[0])
            if not ((ring[:, 0] > -12) & (ring[:, 0] < 32) & (ring[:, 1] > 34) & (ring[:, 1] < 62)).any():
                continue
            x, y = laea(ring[:, 0], ring[:, 1])
            q = _clip_rect(np.c_[x, y], X0, X1, Y0, Y1)
            if len(q) < 3:
                continue
            if k in values:
                val = values[k]
                col = zero_color if (isinstance(norm, colors.Normalize) and not isinstance(norm, colors.BoundaryNorm) and val <= 1e-6 and vmin == 0) else cmap(norm(val))
                ec, lw, zo = '#6B7785', 0.42, 2
            else:
                col, ec, lw, zo = '#FFFFFF', '#CBD5E1', 0.28, 1
            a.add_patch(Polygon(q, facecolor=col, edgecolor=ec, lw=lw, zorder=zo))

    if frame:
        from matplotlib.patches import Rectangle
        a.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec='#CBD5E1', lw=0.4, zorder=4))
    a.set_xlim(X0, X1)
    a.set_ylim(Y0, Y1)
    a.set_aspect('equal')
    a.axis('off')
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)


def export(f, name: str):
    f.canvas.draw()
    r = f.canvas.get_renderer()
    w, h = f.canvas.get_width_height()
    outside, small = [], []
    for t in f.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < 6.99:
            small.append((t.get_text(), t.get_fontsize()))
        b = t.get_window_extent(r)
        if b.width > 0 and b.height > 0 and (b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > w + 0.5 or b.y1 > h + 0.5):
            if t in f.texts:
                outside.append(t.get_text())
            elif t.axes is not None:
                if not t.get_clip_on() or t in (t.axes.xaxis.label, t.axes.yaxis.label):
                    outside.append(t.get_text())
                elif t in t.axes.get_xticklabels():
                    tx = t.get_position()[0]
                    xlo, xhi = sorted(t.axes.get_xlim())
                    if xlo - 1e-6 <= tx <= xhi + 1e-6:
                        outside.append(t.get_text())
                elif t in t.axes.get_yticklabels():
                    ty = t.get_position()[1]
                    ylo, yhi = sorted(t.axes.get_ylim())
                    if ylo - 1e-6 <= ty <= yhi + 1e-6:
                        outside.append(t.get_text())
    for ext in ['pdf', 'svg']:
        f.savefig(FIG / f'{name}.{ext}')
    f.savefig(FIG / f'{name}.png', dpi=240)
    f.savefig(FIG / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})

    from PIL import Image
    Image.open(FIG / f'{name}.png').convert('L').save(QA / f'{name}_gray.png')
    (QA / f'{name}_qa.json').write_text(json.dumps({
        'figure': name,
        'outside_text': outside,
        'text_below_7pt': small,
        'panels': dict(PANELS),
        'sources': dict(SOURCES),
        'width_mm': 180,
        'height_mm': round(f.get_size_inches()[1] / MM, 2),
        'uncertainty': 'Equal-weight weather means; ranges denote scenario min/max across weather years.',
        'passed': not outside and not small,
    }, ensure_ascii=False, indent=2))
    print(f'{name}: outside={len(outside)}, below7pt={len(small)}', flush=True)
    if outside:
        print('  OUTSIDE:', outside)
    if small:
        print('  BELOW 7pt:', small)
    PANELS.clear()
    SOURCES.clear()
    plt.close(f)
