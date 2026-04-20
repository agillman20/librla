#!/usr/bin/env bash
# Re-center subfigure (a) in imageEX.png horizontally without regenerating the
# original plots. The source PNG is a single composite (3142x4086) with three
# rows: row 1 holds (a) alone (left-aligned in the original); rows 2-3 hold
# (b)(c) and (d)(e). We extract panel (a), wipe its old footprint with white,
# and paste it back centered.
#
# Bounding box of panel (a) incl. the "(a) Original" title was found via:
#   magick imageEX.png -crop 3142x1362+0+0 +repage \
#     -bordercolor white -border 1 -fuzz 5% -trim -format "%wx%h%O" info:
# -> 1475x1091+61+146  (i.e. width x height + xoff + yoff)
# Centered x offset = (3142 - 1475) / 2 = 833.5 -> 834.
# Wipe rectangle is enlarged by ~20 px on each side to absorb anti-aliased
# edge pixels that -trim's fuzz tolerance leaves behind.
#
# Usage: scripts/center_imageEX_panel_a.sh [src.png] [dst.png]
# Defaults: imageEX.png -> imageEX_centered.png

set -euo pipefail

SRC="${1:-imageEX.png}"
DST="${2:-imageEX_centered.png}"

PANEL_W=1475
PANEL_H=1091
SRC_X=61
SRC_Y=146
DST_X=834

WIPE_X1=$((SRC_X - 21))           # 40
WIPE_Y1=$((SRC_Y - 16))           # 130
WIPE_X2=$((SRC_X + PANEL_W + 24)) # 1560
WIPE_Y2=$((SRC_Y + PANEL_H + 23)) # 1260

TMP_PANEL="$(mktemp -t a_panel.XXXXXX.png)"
TMP_CLEAN="$(mktemp -t imageEX_clean.XXXXXX.png)"
trap 'rm -f "$TMP_PANEL" "$TMP_CLEAN"' EXIT

magick "$SRC" -crop "${PANEL_W}x${PANEL_H}+${SRC_X}+${SRC_Y}" +repage "$TMP_PANEL"
magick "$SRC" -fill white -draw "rectangle ${WIPE_X1},${WIPE_Y1} ${WIPE_X2},${WIPE_Y2}" "$TMP_CLEAN"
magick "$TMP_CLEAN" "$TMP_PANEL" -geometry "+${DST_X}+${SRC_Y}" -composite "$DST"

echo "wrote $DST"
