#!/usr/bin/env bash

set -euo pipefail

# Required release archives. Each also needs a .sha256 checksum.
REQUIRED_ASSETS=(
  firrtl-bin-linux-x64.tar.gz
  firrtl-bin-macos-x64.tar.gz
  firrtl-bin-macos-arm64.tar.gz
  firrtl-bin-windows-x64.zip
)

usage() {
  echo "Usage: $0 [-h] <firtool version or release tag>"
}

while getopts ':h' OPT; do
  case "$OPT" in
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 1 ;;
  esac
done
shift "$((OPTIND - 1))"

if [[ $# -ne 1 || -z "$1" ]]; then
  usage >&2
  exit 1
fi

# Strip firtool- prefix from argument
VERSION=${1#firtool-}

TAG="firtool-$VERSION"
RELEASE_URL="https://api.github.com/repos/llvm/circt/releases/tags/$TAG"
if ! ASSET_NAMES=$(curl -fsS "$RELEASE_URL" | jq -r '.assets[] | select(.state == "uploaded") | .name'); then
  echo "Could not list firtool release assets: $RELEASE_URL" >&2
  exit 1
fi

# Wait until all supported platforms and their checksums have been uploaded.
for ASSET in "${REQUIRED_ASSETS[@]}"; do
  for SUFFIX in "" .sha256; do
    if ! grep -Fxq -- "$ASSET$SUFFIX" <<< "$ASSET_NAMES"; then
      echo "Required firtool release asset is unavailable: https://github.com/llvm/circt/releases/download/$TAG/$ASSET$SUFFIX" >&2
      exit 1
    fi
  done
done
