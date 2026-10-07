# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fonttools>=4.47",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "scikit-image>=0.22",
# ]
# ///

import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skeletonize import glyph_outline, sources

def write_overlay(path, glyphs, upm, ascent, descent):
    cell, cols, pad = 120, 16, 14
    s = cell / (ascent - descent); rows = max(1, math.ceil(len(glyphs) / cols))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols * cell}" height="{rows * (cell + pad)}" 'f'viewBox="0 0 {cols * cell} {rows * (cell + pad)}" style="background:#fff">']
    for n, (cp, contours, strokes) in enumerate(glyphs):
        ox, oy = (n % cols) * cell, (n // cols) * (cell + pad)
        tx = lambda x, y: f'{ox + x * s:.1f},{oy + (ascent - y) * s:.1f}'
        d = ' '.join('M' + ' L'.join(tx(x, y) for x, y in c) + ' Z' for c in contours)
        out.append(f'<path d="{d}" fill="#000" fill-rule="nonzero"/>')
        for U in strokes:
            out.append(f'<polyline points="{" ".join(tx(x, y) for x, y in U)}" fill="none" stroke="#3d8bff" '
                       f'stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>')
            for x, y in (U[0], U[-1]):
                out.append(f'<circle cx="{tx(x, y).split(",")[0]}" cy="{tx(x, y).split(",")[1]}" r="2" fill="#3d8bff"/>')
        out.append(f'<text x="{ox + 3}" y="{oy + cell + pad - 3}" font-size="10" font-family="monospace" fill="#555">'
                   f'U+{cp:04X} {len(strokes)}</text>')
    out.append('</svg>')
    open(path, 'w', encoding='utf-8').write('\n'.join(out))


def main():
    ap = argparse.ArgumentParser(description='a skeleton over its font\'s glyphs')
    ap.add_argument('skeleton'); ap.add_argument('out')
    ap.add_argument('--font'); ap.add_argument('--fallback'); ap.add_argument('--instance')
    a = ap.parse_args()
    sk = json.load(open(a.skeleton, encoding='utf-8'))
    srcs, meta = sources(a.font or sk['source'], a.fallback or sk.get('fallback'), a.instance or sk.get('instance'))
    glyphs = []
    for key, e in sorted(sk['glyphs'].items(), key=lambda kv: int(kv[0][2:], 16)):
        cp = int(key[2:], 16)
        if not e['strokes']:
            continue
        g = glyph_outline(srcs, cp)
        glyphs.append((cp, g[0] if g else [], [np.asarray(U, float) for U in e['strokes']]))
    write_overlay(a.out, glyphs, sk['upm'], sk['ascent'], sk['descent'])
    print(f'{a.out}: {len(glyphs)} glyphs', file=sys.stderr)

if __name__ == '__main__':
    main()
