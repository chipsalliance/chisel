#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
to-fir $1
(cd generated; $DIR/fir2flo.sh $1)
(cd generated; $DIR/flo2app $1)
