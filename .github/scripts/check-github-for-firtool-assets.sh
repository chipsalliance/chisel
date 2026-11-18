#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "Usage: $0 <firtool version or release tag>" >&2
  exit 1
fi

# Strip firtool- prefix from argument
VERSION=${1#firtool-}

BASE_URL="https://github.com/llvm/circt/releases/download/firtool-$VERSION"

# Wait until all supported platforms and their checksums have been uploaded.
for ASSET in \
  firrtl-bin-linux-x64.tar.gz \
  firrtl-bin-macos-x64.tar.gz \
  firrtl-bin-macos-arm64.tar.gz \
  firrtl-bin-windows-x64.zip; do
  for SUFFIX in "" .sha256; do
    URL="$BASE_URL/$ASSET$SUFFIX"
    if ! curl -o /dev/null -sfIL "$URL"; then
      echo "Required firtool release asset is unavailable: $URL" >&2
      exit 1
    fi
  done
done
