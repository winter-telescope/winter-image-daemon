#!/usr/bin/env bash
set -euo pipefail

# Locate and init conda (adjust if conda lives elsewhere)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh; edit this script to your conda path." >&2
  exit 1
fi

# Activate the repo-local environment
conda activate /home/winter/GIT/winter-image-daemon/.conda

# cd to repo root (script/../..)  <-- important now that the script is in services/linux/
REPO_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_DIR"

# Launch the daemon; pass through all CLI args
exec imagedaemon-daemon "$@"
