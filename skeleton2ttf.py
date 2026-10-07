# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fonttools>=4.47",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "scikit-image>=0.22",
#     "shapely>=2.0",
# ]
# ///

import argparse, hashlib, json, math, os, sys
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

EM_PX = 23.0
THICK_MEAN, THICK_STROKE_SD, THICK_SD, THICK_CORR = 0.94, 0.114, 0.185, 1.896
BIAS_SD, BIAS_CORR = 0.079, 0.9672
THICK_MIN, THICK_MAX, BIAS_MAX = 0.098, 1.21, 1.01
BREAK_LEN = 8.73
BREAK2_MID, BREAK2_SCALE = 27.115, 3.979
BREAK2_GAP = 8.016
BREAK_WIDTH_MU, BREAK_WIDTH_SIGMA = 1.281, 0.3115
BREAK_FACTOR_MEAN, BREAK_FACTOR_SD, BREAK_FACTOR_MAX = 0.421, 0.121, 0.6516
BREAK_BIAS_SD = 0.1494
BREAK_POS_EDGES = [0, 0.101, 0.25, 0.5, 0.75, 0.9009, 1.0]
BREAK_POS_COUNTS = [3, 87, 86, 38, 13, 3]

STEP = 0.5
SEED_GRID = 64
REDRAWS = 4
PEN = 7.8 / 109
TURN = 30.0

def parse_charset(path):
    return sorted({ord(c) for c in open(path, encoding='utf-8').read() if not c.isspace()})

def resample(P, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if d[-1] < 1e-9:
        return P[:1].copy()
    t = np.linspace(0, d[-1], max(2, int(round(d[-1] / step)) + 1))
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)

def ou(n, step, mean, sd, corr, rng):
    a = math.exp(-step / corr); x = np.empty(n); x[0] = mean + sd * rng.standard_normal()
    for k in range(1, n):
        x[k] = mean + (x[k - 1] - mean) * a + sd * math.sqrt(1 - a * a) * rng.standard_normal()
    return x

def breakages(L, rng):
    out = []
    if rng.random() >= 1 - math.exp(-L / BREAK_LEN):
        return out
    p = np.array(BREAK_POS_COUNTS, float); p /= p.sum()
    def one():
        i = rng.choice(len(p), p=p)
        return (
            L * rng.uniform(BREAK_POS_EDGES[i], BREAK_POS_EDGES[i + 1]),
            min(float(rng.lognormal(BREAK_WIDTH_MU, BREAK_WIDTH_SIGMA)), L) / 2,
            float(np.clip(rng.normal(BREAK_FACTOR_MEAN, BREAK_FACTOR_SD), THICK_MIN, BREAK_FACTOR_MAX)),
            float(rng.normal(0, BREAK_BIAS_SD))
        )
    out.append(one())
    if rng.random() < 1 / (1 + math.exp(-(L - BREAK2_MID) / BREAK2_SCALE)):
        for _ in range(50):
            b = one()
            if abs(b[0] - out[0][0]) >= BREAK2_GAP:
                out.append(b); break
    return out


def scuff_stroke(n, rng, thickness):
    s = np.arange(n) * STEP
    w = ou(n, STEP, rng.normal(THICK_MEAN, THICK_STROKE_SD), THICK_SD, THICK_CORR, rng)
    o = ou(n, STEP, 0.0, BIAS_SD, BIAS_CORR, rng)
    bs = breakages(float(s[-1]), rng)
    for c, h, f, u in bs:
        lam = np.clip(1 - np.abs(s - c) / max(h, 1e-6), 0, 1)
        w = w * (1 - (1 - f) * lam); o = (1 - lam) * o + lam * u
    return np.clip(w * thickness, THICK_MIN, THICK_MAX), np.clip(o, -BIAS_MAX, BIAS_MAX), len(bs)


def stroke_rng(seed, cp, stroke, upm, ascent):
    x = int(math.floor(stroke[:, 0].min() * SEED_GRID / upm))
    y = int(math.floor((ascent - stroke[:, 1].max()) * SEED_GRID / upm))
    h = hashlib.sha256(f'{seed}:{cp}:{x}:{y}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], 'little'))


def stroke_shape(P, w, o, radius):
    if len(P) == 1:
        return Point(P[0] + 0).buffer(w[0] * radius, quad_segs=8)
    k = max(1, int(round(2.0 / STEP))); idx = np.arange(len(P))
    T = P[np.minimum(idx + k, len(P) - 1)] - P[np.maximum(idx - k, 0)]
    T /= np.maximum(np.hypot(*T.T), 1e-9)[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    C = P + o[:, None] * N; r = w * radius
    L, R = C + r[:, None] * N, C - r[:, None] * N
    parts = [Point(C[0]).buffer(r[0], quad_segs=8), Point(C[-1]).buffer(r[-1], quad_segs=8)]
    ang = np.degrees(np.arctan2(T[:, 1], T[:, 0]))
    for i in range(1, len(C) - 1):
        a, b = max(0, i - 2), min(len(C) - 1, i + 2)
        if abs((ang[b] - ang[a] + 180) % 360 - 180) >= TURN:
            parts.append(Point(C[i]).buffer(r[i], quad_segs=8))
    for i in range(len(C) - 1):
        q = Polygon([L[i], L[i + 1], R[i + 1], R[i]]).buffer(0)
        if not q.is_empty:
            parts.append(q)
    return unary_union(parts)


def glyph(strokes, cp, upm, ascent, seed, thickness, em_px):
    k = em_px / upm
    paths = [resample(np.stack([U[:, 0] * k, -U[:, 1] * k], 1), STEP) if len(U) > 1 else np.stack([U[:, 0] * k, -U[:, 1] * k], 1) for U in strokes]
    for redraw in range(REDRAWS + 1):
        draws = [scuff_stroke(len(S), stroke_rng(seed + redraw, cp, U, upm, ascent), 1.0) for S, U in zip(paths, strokes)]
        if sum(d[2] for d in draws) > 0:
            break
    radius = PEN * em_px * thickness / 2
    geom = unary_union([stroke_shape(S, w, o, radius) for S, (w, o, _) in zip(paths, draws)]).simplify(0.02)
    polys = [geom] if geom.geom_type == 'Polygon' else list(getattr(geom, 'geoms', []))
    pen = TTGlyphPen(None)
    for poly in polys:
        if poly.is_empty or poly.geom_type != 'Polygon':
            continue
        poly = orient(poly, sign=1.0)
        for ring in [poly.exterior] + list(poly.interiors):
            pts = [(int(round(x / k)), int(round(-y / k))) for x, y in list(ring.coords)[:-1]]
            pts = [p for i, p in enumerate(pts) if p != pts[i - 1]]
            if len(pts) < 3:
                continue
            pen.moveTo(pts[0])
            for p in pts[1:]:
                pen.lineTo(p)
            pen.closePath()
    return pen.glyph()


def main():
    ap = argparse.ArgumentParser(description='skeleton -> scuffy font')
    ap.add_argument('skeleton')
    ap.add_argument('out')
    ap.add_argument('--charset')
    ap.add_argument('--family')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--thickness', type=float, default=1.0)
    ap.add_argument('--em-px', type=float, default=EM_PX)
    ap.add_argument('--cell', type=int)
    a = ap.parse_args()
    sk = json.load(open(a.skeleton, encoding='utf-8'))
    upm, ascent, descent = sk['upm'], sk['ascent'], sk['descent']
    entries = {int(key[2:], 16): e for key, e in sk['glyphs'].items()}
    if a.charset:
        want = set(parse_charset(a.charset)) | {0x20, 0x3000}
        entries = {cp: e for cp, e in entries.items() if cp in want}
    family = a.family or os.path.splitext(os.path.basename(a.out))[0]
    cell = a.cell or upm
    order = ['.notdef'] + [f'uni{cp:04X}' for cp in sorted(entries)]
    glyphs = {'.notdef': TTGlyphPen(None).glyph()}; metrics = {'.notdef': (cell, 0)}
    for cp in sorted(entries):
        e = entries[cp]; name = f'uni{cp:04X}'
        dx = (cell - e['advance']) / 2
        strokes = [np.asarray(U, float) + (dx, 0) for U in e['strokes']]
        g = glyph(strokes, cp, upm, ascent, a.seed, a.thickness, a.em_px) if strokes else TTGlyphPen(None).glyph()
        glyphs[name] = g
        metrics[name] = (cell, 0)
    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({cp: f'uni{cp:04X}' for cp in entries})
    fb.setupGlyf(glyphs)
    glyf = fb.font['glyf']
    for name in order:
        g = glyf[name]; g.recalcBounds(glyf)
        metrics[name] = (metrics[name][0], getattr(g, 'xMin', 0))
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupNameTable(dict(familyName=family, styleName='Regular', uniqueFontIdentifier=f'{family}-Regular;scuffed', fullName=f'{family} Regular', psName=f'{family}-Regular'.replace(' ', ''), version='Version 1.000'))
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent, sTypoLineGap=0, usWinAscent=ascent, usWinDescent=-descent, achVendID='NONE', xAvgCharWidth=cell)
    fb.font['OS/2'].panose.bFamilyType = 2; fb.font['OS/2'].panose.bProportion = 9   # monospaced
    fb.setupPost(isFixedPitch=1)
    fb.save(a.out)
    print(f'{a.out}: {len(entries)} glyphs ({sum(len(e["strokes"]) for e in entries.values())} strokes) from ' f'{os.path.basename(a.skeleton)}', file=sys.stderr)

if __name__ == '__main__':
    main()
