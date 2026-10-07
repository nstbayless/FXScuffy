# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fonttools>=4.47",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "scikit-image>=0.22",
# ]
# ///

import argparse, json, math, sys
import numpy as np
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont
from scipy import ndimage
from skimage.draw import polygon as fill_polygon
from skimage.morphology import skeletonize

EM_PX = 23.0
RASTER = 128
STEP = 0.5
SMOOTH = 1.0
CULL_SPUR = 1.5
JOIN_DEG = 35.0
CORNER_DEG = 60.0
CORNER_PX = 3.0
SPACES = [0x20, 0x3000]

def parse_charset(path):
    return sorted({ord(c) for c in open(path, encoding='utf-8').read() if not c.isspace()})

def outline(glyphset, name):
    pen = DecomposingRecordingPen(glyphset)
    glyphset[name].draw(pen)
    contours, cur = [], []
    for op, args in pen.value:
        if op == 'moveTo':
            cur = [args[0]]
        elif op == 'lineTo':
            cur.append(args[0])
        elif op == 'qCurveTo':
            pts = list(args)
            if pts[-1] is None:
                pts = pts[:-1]
                start = ((pts[-1][0] + pts[0][0]) / 2, (pts[-1][1] + pts[0][1]) / 2)
                cur = [start]; pts = pts + [start]
            p0 = cur[-1]
            for i, c in enumerate(pts[:-1]):
                end = pts[-1] if i == len(pts) - 2 else ((c[0] + pts[i + 1][0]) / 2, (c[1] + pts[i + 1][1]) / 2)
                for t in np.linspace(0, 1, 9)[1:]:
                    cur.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * end[0], (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * end[1]))
                p0 = end
        elif op == 'curveTo':
            p0 = cur[-1]; c1, c2, end = args
            for t in np.linspace(0, 1, 13)[1:]:
                u = 1 - t
                cur.append((u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * end[0], u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * end[1]))
        elif op in ('closePath', 'endPath'):
            if len(cur) > 2:
                contours.append(np.array(cur, float))
            cur = []
    return contours

def rasterize(contours, scale, origin, size):
    acc = np.zeros((size, size), int)
    for c in contours:
        x = (c[:, 0] - origin[0]) * scale; y = (origin[1] - c[:, 1]) * scale
        area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]
        rr, cc = fill_polygon(y, x, (size, size))
        acc[rr, cc] += 1 if area > 0 else -1
    return acc != 0

def skeleton_paths(mask):
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    on = set(zip(ys.tolist(), xs.tolist()))
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in on]
    deg = {p: len(nb(p)) for p in on}
    junc = np.zeros(mask.shape, bool)
    for p, d in deg.items():
        if d != 2:
            junc[p] = True
    lab, _ = ndimage.label(junc, structure=np.ones((3, 3)))
    node = {p: int(lab[p]) for p in on if lab[p]}
    paths, done = [], set()
    for start in [p for p in on if p in node]:
        for q in nb(start):
            if q in node and node[q] == node[start]:
                continue
            if (start, q) in done:
                continue
            path, prev, cur = [start], start, q
            while True:
                path.append(cur)
                if cur in node:
                    break
                nxt = [r for r in nb(cur) if r != prev and r not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            done.add((start, path[1])); done.add((path[-1], path[-2]))
            paths.append(dict(pts=np.array(path, float), a=node.get(path[0]), b=node.get(path[-1])))
    touched = {n for pa in paths for n in (pa['a'], pa['b'])}
    for n in set(node.values()) - touched:
        cl = np.array([p for p, m in node.items() if m == n], float)
        paths.append(dict(pts=cl.mean(0, keepdims=True), a=None, b=None))
    seen = {tuple(map(int, p)) for pa in paths for p in pa['pts']}
    for p in on:
        if p in seen or p in node:
            continue
        loop, prev, cur = [p], None, p
        while True:
            nxt = [r for r in nb(cur) if r != prev and r not in loop[-2:]]
            if not nxt or nxt[0] == p:
                break
            prev, cur = cur, nxt[0]; loop.append(cur)
        seen.update(loop)
        paths.append(dict(pts=np.array(loop + [p], float), a=None, b=None))
    return paths


def strokes_from(paths, dist, rpx):
    def length(P):
        return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0
    def radius(P, i):
        y, x = P[i].astype(int); return float(dist[y, x])
    changed = True
    while changed:
        changed = False
        count = {}
        for p in paths:
            for n in (p['a'], p['b']):
                if n is not None:
                    count[n] = count.get(n, 0) + 1
        for p in list(paths):
            for end, other in (('a', 'b'), ('b', 'a')):
                n, m = p[end], p[other]
                if n is not None and count.get(n, 0) >= 3 and (m is None or count.get(m, 0) == 1):
                    r = radius(p['pts'], 0 if end == 'a' else -1)
                    if length(p['pts']) < CULL_SPUR * r:
                        paths = [q for q in paths if q is not p]; changed = True; break
            if changed:
                break
    def leaving(p, end, reach):
        P = p['pts'] if end == 'a' else p['pts'][::-1]
        d = np.cumsum(np.r_[0, np.hypot(*np.diff(P, axis=0).T)]); j = min(np.searchsorted(d, reach), len(P) - 1)
        v = P[j] - P[0]; n = np.hypot(*v)
        return v / n if n > 1e-9 else np.array([1.0, 0.0])
    ends = {}
    for i, p in enumerate(paths):
        for end in ('a', 'b'):
            if p[end] is not None:
                ends.setdefault(p[end], []).append((i, end))
    link = {}
    for n, es in ends.items():
        reach = 1.5 * max(radius(paths[i]['pts'], 0 if e == 'a' else -1) for i, e in es) + 2
        cand = []
        for x in range(len(es)):
            for y in range(x + 1, len(es)):
                (i, ei), (j, ej) = es[x], es[y]
                if i == j:
                    continue
                c = float(-leaving(paths[i], ei, reach) @ leaving(paths[j], ej, reach))
                cand.append((math.degrees(math.acos(max(-1.0, min(1.0, c)))), es[x], es[y]))
        used = set()
        two = len(es) == 2
        for turn, u, v in sorted(cand):
            if u in used or v in used or (turn > JOIN_DEG and not two):
                continue
            used |= {u, v}; link[u] = v; link[v] = u
    strokes, taken = [], set()
    for i in range(len(paths)):
        if i in taken:
            continue
        start_end = 'a' if (i, 'a') not in link else ('b' if (i, 'b') not in link else 'a')
        seq, k, e = [], i, start_end
        while k not in taken:
            taken.add(k)
            P = paths[k]['pts'] if e == 'a' else paths[k]['pts'][::-1]
            seq.append(P if not seq else P[1:])
            nxt = link.get((k, 'b' if e == 'a' else 'a'))
            if nxt is None:
                break
            k, e = nxt
        strokes.append(np.vstack(seq) * rpx)
    return strokes


def resample(P, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    if d[-1] < 1e-9:
        return P[:1].copy()
    t = np.linspace(0, d[-1], max(2, int(round(d[-1] / step)) + 1))
    return np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)


def smooth(P, sigma):
    if len(P) < 5:
        return P
    Q = ndimage.gaussian_filter1d(P, sigma, axis=0, mode='nearest')
    Q[0], Q[-1] = P[0], P[-1]
    return Q


def split_corners(P, step):
    k = max(1, int(round(CORNER_PX / step))); n = len(P)
    if n < 2 * k + 1:
        return [P]
    closed = np.hypot(*(P[0] - P[-1])) < 2 * step
    turn = np.zeros(n)
    for i in range(k, n - k):
        a = P[i] - P[i - k]; b = P[i + k] - P[i]
        c = a @ b / max(np.hypot(*a) * np.hypot(*b), 1e-12)
        turn[i] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    cut = []
    for i in np.argsort(-turn):
        if turn[i] < CORNER_DEG:
            break
        if all(abs(i - j) > 2 * k for j in cut):
            cut.append(int(i))
    if not cut or not closed:
        return [P]
    cut.sort()
    if closed:
        Q = np.vstack([P[cut[0]:-1], P[:cut[0] + 1]]); cut = [c - cut[0] for c in cut] + [len(Q) - 1]
        return [Q[a:b + 1] for a, b in zip(cut, cut[1:]) if b > a]
    cut = [0] + cut + [n - 1]
    return [P[a:b + 1] for a, b in zip(cut, cut[1:]) if b > a]


def skeleton(contours, upm):
    pts = np.vstack(contours)
    lo = pts.min(0) - upm * 0.1; hi = pts.max(0) + upm * 0.1
    scale = RASTER / upm; size = int(math.ceil(max(hi - lo) * scale)) + 2
    mask = rasterize(contours, scale, (lo[0], hi[1]), size)
    if mask.sum() < 4:
        return []
    dist = ndimage.distance_transform_edt(mask)
    px = EM_PX / upm
    out = []
    for S in strokes_from(skeleton_paths(mask), dist, 1.0):
        U = np.stack([S[:, 1] / scale + lo[0], hi[1] - S[:, 0] / scale], 1) * px
        if len(U) > 1:
            U = resample(smooth(resample(U, STEP), SMOOTH / STEP), STEP)
        out += [V / px for V in split_corners(U, STEP)]
    return out


def open_font(path, instance=None):
    f = TTFont(path); location = None
    if 'fvar' in f:
        want = instance or 'Regular'
        inst = [i for i in f['fvar'].instances if f['name'].getDebugName(i.subfamilyNameID) == want]
        if not inst and instance:
            sys.exit(f'{path}: no instance {instance!r}')
        location = inst[0].coordinates if inst else None
    return f, f.getGlyphSet(location=location), f.getBestCmap()


def sources(path, fallback=None, instance=None):
    fonts = [open_font(p, instance) for p in [path] + ([fallback] if fallback else [])]
    upm = fonts[0][0]['head'].unitsPerEm
    return ([(gs, cm, upm / f['head'].unitsPerEm, f['hmtx']) for f, gs, cm in fonts],
            dict(font=fonts[0][0]['name'].getDebugName(4), upm=upm, ascent=fonts[0][0]['hhea'].ascent, descent=fonts[0][0]['hhea'].descent))


def glyph_outline(srcs, cp):
    for gs, cm, sc, hmtx in srcs:
        if cp in cm:
            return [c * sc for c in outline(gs, cm[cp])], round(hmtx[cm[cp]][0] * sc)
    return None


def main():
    ap = argparse.ArgumentParser(description='a font\'s glyphs as strokes')
    ap.add_argument('font'); ap.add_argument('charset'); ap.add_argument('out')
    ap.add_argument('--fallback', help='a font for the characters FONT lacks')
    ap.add_argument('--instance', help='named instance of a variable font (default: Regular, if it has one)')
    a = ap.parse_args()
    srcs, meta = sources(a.font, a.fallback, a.instance)
    glyphs = {}
    todo = parse_charset(a.charset)
    for k, cp in enumerate(todo):
        g = glyph_outline(srcs, cp)
        if g is None:
            continue
        contours, advance = g
        strokes = skeleton(contours, meta['upm']) if contours else []
        glyphs[f'U+{cp:04X}'] = dict(advance=advance, strokes=[[[round(float(x), 1), round(float(y), 1)] for x, y in U] for U in strokes])
        if (k + 1) % 100 == 0:
            print(f'{k + 1}/{len(todo)}', file=sys.stderr)
    missing = [cp for cp in todo if f'U+{cp:04X}' not in glyphs]
    for cp in SPACES:
        g = glyph_outline(srcs, cp)
        if g and f'U+{cp:04X}' not in glyphs:
            glyphs[f'U+{cp:04X}'] = dict(advance=g[1], strokes=[])
    json.dump(dict(source=a.font, fallback=a.fallback, instance=a.instance, **meta, glyphs=glyphs), open(a.out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'{a.out}: {len(glyphs)} glyphs, {sum(len(g["strokes"]) for g in glyphs.values())} strokes' + (f'; not in the font: {"".join(chr(c) for c in missing)}' if missing else ''), file=sys.stderr)

if __name__ == '__main__':
    main()
