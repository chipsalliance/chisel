# Installs stanza into /usr/local/bin
# TODO Talk to Patrick to fill this in
root_dir ?= $(PWD)
test_src_dir ?= $(root_dir)/test/unit
test_out_dir ?= $(root_dir)/test/unit/out
firrtl_dir ?= $(root_dir)/src/main/stanza

all: build check

install-stanza:



build: 
	cd $(firrtl_dir) && stanzam -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl

# Runs single test
check: 
	./firrtl $(test_src_dir)/gcd.fir | tee $(test_out_dir)/gcd.out

diff:
	diff test/unit/out/* test/unit/cor/*
