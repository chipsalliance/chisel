set -e
# Install Yosys (https://github.com/cliffordwolf/yosys)
if [ ! -f $INSTALL_DIR/bin/yosys ]; then
  mkdir -p $INSTALL_DIR
  git clone https://github.com/cliffordwolf/yosys.git
  cd yosys
  git pull
  git checkout yosys-0.7
  # Workaround moving ABC repo
  git apply ../.fix_yosys_abc.patch
  make
  make PREFIX=$INSTALL_DIR install
fi
