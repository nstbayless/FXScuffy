#!/usr/bin/env bash
# usage: ./build.sh [--skeletons] [--seed N] [jpn kor chi cht]
set -euo pipefail
cd "$(dirname "$0")"

skeletons=0; seed=0; langs=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skeletons) skeletons=1 ;;
        --seed) seed="$2"; shift ;;
        *) langs+=("$1") ;;
    esac
    shift
done
[ ${#langs[@]} -gt 0 ] || langs=(jpn kor chi cht)

source_of() {
    case "$1" in
        jpn) echo "KosugiMaru-Regular.ttf - KosugiMaru-LICENSE.txt" ;;
        kor) echo "GowunBatang-Regular.ttf - Gowun_Batang_LICENSE.txt" ;;
        chi) echo "ResourceHanRoundedCN-VF.otf ResourceHanRoundedHK-VF ResourceHanRounded_LICENSE.txt" ;;
        cht) echo "ResourceHanRoundedTW-VF.otf ResourceHanRoundedCN-VF.otf ResourceHanRounded_LICENSE.txt" ;;
        *) echo "unknown glyph set: $1" >&2; exit 1 ;;
    esac
}

mkdir -p fonts skeletons
for lang in "${langs[@]}"; do
    echo "$lang..."
    read -r font fallback licence <<< "$(source_of "$lang")"
    if [ "$skeletons" = 1 ]; then
        fb=(); [ "$fallback" = - ] || fb=(--fallback "sources/$fallback")
        uv run --quiet skeletonize.py "sources/$font" "glyphs/$lang.txt" "skeletons/$lang.json" "${fb[@]}"
        uv run --quiet overlay.py "skeletons/$lang.json" "skeletons/$lang.svg"
    fi
    uv run --quiet skeleton2ttf.py "skeletons/$lang.json" "fonts/fxscuffy-$lang.ttf" --seed "$seed"
    cp "sources/$licence" fonts/
done
