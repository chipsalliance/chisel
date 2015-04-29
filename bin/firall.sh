#!/bin/bash
shift 1
for file in "$@"; do
    sbt "run $file"
done
