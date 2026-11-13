#!/usr/bin/env bash
# Pull ultrasound videos from the LRZ cluster to the local mac over SSH.
#
# Usage:
#   bash scripts/pull_cluster_videos.sh
#
# Why a file (not a pasted here-block):
#   Pasting a multi-line "( ... )" block with inline "#" comments into an
#   interactive zsh can make zsh try to run the "#" as a command
#   ("command not found: #"). Running a saved script with bash avoids that.

set -euo pipefail

SSH_HOST="${SSH_HOST:-ai}"
REMOTE="${REMOTE:-/dss/mcmlscratch/04/ge75vid2/haoyu/live-ultrasound-video-understanding/cluster_data/videos}"
DEST="${DEST:-$HOME/Desktop/Semester 3/live-ultrasound-video-understanding/cluster_data/videos_from_ssh}"

# Keep this many GiB free on the local disk after accounting for the copy.
RESERVE_GIB="${RESERVE_GIB:-15}"

mkdir -p "$DEST"

# Check remote size; reserve RESERVE_GIB locally.
REMOTE_KB=$(ssh -o ConnectTimeout=15 "$SSH_HOST" "du -sk '$REMOTE'" | awk '{print $1}')
FREE_KB=$(df -k "$DEST" | awk 'END {print $4}')
RESERVE_KB=$(( RESERVE_GIB * 1024 * 1024 ))

if [ "$REMOTE_KB" -gt "$(( FREE_KB - RESERVE_KB ))" ]; then
  echo "Not enough local free space to copy and keep ${RESERVE_GIB} GiB free. Aborting."
  echo "  remote=${REMOTE_KB} KB  free=${FREE_KB} KB  reserve=${RESERVE_KB} KB"
  exit 1
fi

rsync -avh --partial --progress \
  --include='*/' \
  --include='*.mp4' --include='*.MP4' \
  --include='*.mkv' --include='*.webm' --include='*.mov' \
  --exclude='*' \
  "$SSH_HOST:$REMOTE/" "$DEST/"

echo "Download complete: $DEST"
