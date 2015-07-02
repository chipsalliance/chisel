#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
shift 1
for file in "$@"; do
    $DIR/build.sh $file
done
