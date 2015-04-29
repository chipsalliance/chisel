#!/bin/bash

to-fir $1
(cd generated; fir2flo.sh $1)
(cd generated; flo-app $1)
