#!/usr/bin/env bash
# Regenerate the raster icons in public/ from the SVG sources (needs rsvg-convert and ImageMagick).
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
for size in 16 32 48; do rsvg-convert -w $size -h $size public/favicon.svg -o "$tmp/$size.png"; done
magick "$tmp/16.png" "$tmp/32.png" "$tmp/48.png" public/favicon.ico
rsvg-convert -w 192 -h 192 public/favicon.svg -o "$tmp/192.png"
magick "$tmp/192.png" -quality 90 public/favicon.webp
rsvg-convert -w 1200 -h 630 icons/og.svg -o "$tmp/og.png"
magick "$tmp/og.png" -quality 88 public/og.webp
rm -r "$tmp"
