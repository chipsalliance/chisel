#!/bin/sh
# Parse the output of:
#  sbt "show compile:sources compile:resources".
#
# Set the field delimiter to any of '(', ',', or ')'
#  and split lines beginning with:
#   [info] ArrayBuffer(
#   [info] List(
#  throwing away the initial 'field' up to and including the '('.
gawk -F '\\(|,|\\)' '/\[info\]\s+((ArrayBuffer)|(List))\(/ { $1 = ""; print $0 }'
