# Installs stanza into /usr/local/bin
# TODO Talk to Patrick to fill this in
root_dir ?= $(PWD)
test_dir ?= $(root_dir)/test
firrtl_dir ?= $(root_dir)/src/main/stanza

all: build check

install-stanza:

build: 
	cd $(firrtl_dir) && stanzam -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl

# Runs single test
check: 
	cd $(test_dir) && lit -v . --path=$(root_dir)/utils/bin/
	cat $(test_dir)/unit/gcd.fir.out
