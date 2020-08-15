set -e
# Install Z3 (https://github.com/Z3Prover/z3)
if [ ! -f $INSTALL_DIR/bin/z3 ]; then 
  mkdir -p $INSTALL_DIR
  # download prebuilt binary
  wget https://github.com/Z3Prover/z3/releases/download/z3-4.8.8/z3-4.8.8-x64-ubuntu-16.04.zip
  unzip z3-4.8.8-x64-ubuntu-16.04.zip
  mv ./z3-4.8.8-x64-ubuntu-16.04/bin/z3 $INSTALL_DIR/bin/z3
fi
