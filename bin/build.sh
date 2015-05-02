#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
$DIR/to-fir.sh $1
(cd generated; $DIR/fir2flo.sh $1)
(cd generated; $DIR/flo2app.sh $1)
