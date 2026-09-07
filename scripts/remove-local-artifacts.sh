#!/usr/bin/env bash
# Remove local, reproducible inputs and intermediate training state.
# Generated analysis results in examples/01_difftre/data/*.csv and plots/ stay.

set -euo pipefail

remove=false
pinning_only=false

for argument in "$@"; do
    case "$argument" in
        --remove) remove=true ;;
        --pinning) pinning_only=true ;;
        --help|-h)
            echo "Usage: $0 --remove [--pinning]"
            echo "  --remove   delete disposable local artifacts"
            echo "  --pinning  delete only final interface-pinning outputs"
            exit 0
            ;;
        *)
            echo "Unknown option: $argument" >&2
            exit 2
            ;;
    esac
done

if [[ "$remove" != true ]]; then
    echo "This removes disposable local artifacts. Re-run with --remove."
    exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$pinning_only" == true ]]; then
    # Keep models and prepared inputs so evaluate_pinning.py can be rerun alone.
    targets=("examples/02_diffttc/training/output/final")
else
    targets=(
        "examples/01_difftre/Checkpoints"
        "examples/01_difftre/data/forces.npy"
        "examples/01_difftre/data/positions.npy"
        "examples/02_diffttc/example.py"
    )
fi

for relative_path in "${targets[@]}"; do
    path="$repo_root/$relative_path"
    if [[ -e "$path" ]]; then
        rm -rf -- "$path"
        echo "Removed $relative_path"
    fi
done
