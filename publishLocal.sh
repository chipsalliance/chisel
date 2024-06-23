#!/bin/bash
set -e

SCALA_VERSION="2.13.15"

if [ "$1" == "clean" ]; then
    # git clean -fdxf target project .bps
    sbt "clean; unipublish/clean; ++ ${SCALA_VERSION}! -v; clean; unipublish/clean"
fi
sbt "++ ${SCALA_VERSION}! -v; compile; unipublish/publishLocal"
