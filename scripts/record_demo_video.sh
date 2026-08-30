#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
frontend_dir="$repo_root/app/frontend"
raw_dir="$frontend_dir/test-results-real/demo-video"
output="$repo_root/results/demo-video.mp4"
temporary_output="$repo_root/results/.demo-video-$$.mp4"

command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
test -x "$frontend_dir/node_modules/.bin/playwright"
mkdir -p "$raw_dir" "$repo_root/results"
find "$raw_dir" -mindepth 1 -delete
trap 'rm -f "$temporary_output"' EXIT

"$repo_root/.venv/bin/python" "$repo_root/scripts/release.py" \
  --root "$repo_root" --verify >/dev/null

(
  cd "$frontend_dir"
  ./node_modules/.bin/playwright test --config=playwright.demo.config.ts
)

mapfile -d '' videos < <(find "$raw_dir" -type f -name '*.webm' -print0)
if [[ ${#videos[@]} -ne 1 ]]; then
  echo "Expected one raw Playwright video, found ${#videos[@]}" >&2
  exit 1
fi

raw_duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videos[0]}")"
awk -v duration="$raw_duration" 'BEGIN { exit !(duration >= 242) }' || {
  echo "Raw tour is ${raw_duration}s; at least 242s is required" >&2
  exit 1
}
trim_start="$(awk -v duration="$raw_duration" 'BEGIN { printf "%.6f", duration - 240 }')"

ffmpeg -hide_banner -loglevel error -y -i "${videos[0]}" \
  -vf "trim=start=${trim_start}:duration=240,setpts=PTS-STARTPTS,fps=30,scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2,format=yuv420p" \
  -frames:v 7200 -an -c:v libx264 -preset medium -crf 18 -movflags +faststart "$temporary_output"

final_duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$temporary_output")"
dimensions="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$temporary_output")"
awk -v duration="$final_duration" 'BEGIN { exit !(duration >= 239.9995 && duration <= 240.0005) }' || {
  echo "Encoded duration is ${final_duration}s, expected exactly 240.000s" >&2
  exit 1
}
if [[ "$dimensions" != "1440x900" ]]; then
  echo "Encoded dimensions are ${dimensions}, expected 1440x900" >&2
  exit 1
fi

mv "$temporary_output" "$output"
printf 'Recorded %s (%.3fs, %s)\n' "$output" "$final_duration" "$dimensions"
