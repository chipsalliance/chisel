#!/bin/bash
set -ev
# If you want to run the tests requiring a backend, you'll need to install verilator.
# Install Verilator (http://www.veripool.org/projects/verilator/wiki/Installing)
bash .install_verilator.sh
# Install firrtl
bash .install-firrtl.sh
