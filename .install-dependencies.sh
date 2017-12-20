#!/bin/bash
set -ev
# Install Verilator (http://www.veripool.org/projects/verilator/wiki/Installing)
bash .install_verilator.sh
# Install firrtl
bash .install-firrtl.sh
