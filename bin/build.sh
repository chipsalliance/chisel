#!/bin/bash

sbt "run $1"
(cd generated; fir2flo.sh $1)
(cd generated; flo-app $1)
