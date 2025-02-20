#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
  echo "Error: This script requires exactly 1 argument."
  echo "Usage: $0 <chisel version>"
  exit 1
fi

THISDIR=$(dirname "$0")
VERSION=$1

sed "s/@VERSION@/$VERSION/g" $THISDIR/chisel-example.scala
