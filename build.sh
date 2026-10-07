#!/usr/bin/env bash
# usage: ./build.sh [--skeletons] [--full] [--seed N] [jpn kor chi cht]
# --full: the extended glyph sets (glyphs/jpn-jis0208, chi-gb2312, cht-big5, kor-hangul) 
set -euo pipefail
cd "$(dirname "$0")"

skeletons=0; full=0; seed=0; langs=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skeletons) skeletons=1 ;;
        --full) full=1 ;;
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
        chi) echo "ResourceHanRoundedCN-VF.otf ResourceHanRoundedHK-VF.otf ResourceHanRounded_LICENSE.txt" ;;
        cht) echo "ResourceHanRoundedTW-VF.otf ResourceHanRoundedCN-VF.otf ResourceHanRounded_LICENSE.txt" ;;
        *) echo "unknown glyph set: $1" >&2; exit 1 ;;
    esac
}

full_set() {
    case "$1" in jpn) echo jpn-jis0208 ;; chi) echo chi-gb2312 ;; cht) echo cht-big5 ;; kor) echo kor-hangul ;; esac
}

mkdir -p fonts skeletons
for lang in "${langs[@]}"; do
    echo "$lang..."
    read -r font fallback licence <<< "$(source_of "$lang")"
    set="$lang"; name="$lang"
    [ "$full" = 0 ] || { set="$(full_set "$lang")"; name="$lang-full"; }
    if [ "$skeletons" = 1 ]; then
        fb=(); [ "$fallback" = - ] || fb=(--fallback "sources/$fallback")
        uv run --quiet skeletonize.py "sources/$font" "glyphs/$set.txt" "skeletons/$name.json" "${fb[@]}"
        [ "$full" = 1 ] || uv run --quiet overlay.py "skeletons/$name.json" "skeletons/$name.svg"
    fi
    uv run --quiet skeleton2ttf.py "skeletons/$name.json" "fonts/fxscuffy-$name.ttf" --seed "$seed"
    cp "sources/$licence" fonts/
done
