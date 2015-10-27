root_dir ?= $(PWD)
test_dir ?= $(root_dir)/test
regress_dir ?= $(root_dir)/regress
firrtl_dir ?= $(root_dir)/src/main/stanza
install_dir ?= $(root_dir)/utils/bin

stanza ?= $(install_dir)/stanza

all-noise: 
	${MAKE} all || ${MAKE} fail

all: done

# Installs Stanza into $(insall_dir)
stanza_zip_name = $(subst Darwin,mac,$(subst Linux,linux,$(shell uname)))
stanza_target_name = $(subst Darwin,os-x,$(subst Linux,linux,$(shell uname)))

$(root_dir)/src/lib/stanza/stamp: src/lib/stanza-$(stanza_zip_name).zip
	rm -rf src/lib/stanza
	mkdir -p src/lib
	cd src/lib && unzip stanza-$(stanza_zip_name).zip
	touch $@

$(stanza): $(root_dir)/src/lib/stanza/stamp
	cd src/lib/stanza && ./stanza -platform $(stanza_target_name) -install $(stanza)

build-deploy: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl

build: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl

build-fast: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl -flags OPTIMIZE

build-deploy: 
	cd $(firrtl_dir) && $(stanza) -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl-stanza
	make set-stanza

build: 
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl-stanza
	make set-stanza

build-fast: 
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl-stanza -flags OPTIMIZE
	make set-stanza

check: 
	cd $(test_dir) && lit -v . --path=$(root_dir)/utils/bin/

regress: 
	cd $(regress_dir) && firrtl -i rocket.fir -o rocket.v -X verilog

passes: 
	cd $(test_dir)/passes && lit -v . --path=$(root_dir)/utils/bin/

errors:
	cd $(test_dir)/errors && lit -v . --path=$(root_dir)/utils/bin/

features:
	cd $(test_dir)/features && lit -v . --path=$(root_dir)/utils/bin/

custom:
	cd $(test_dir)/custom && lit -v . --path=$(root_dir)/utils/bin/ --max-time=10

clean:
	rm -f $(test_dir)/*/*/*.out
	rm -f $(test_dir)/*/*.out
	rm -rf src/lib/stanza
	rm -f $(stanza)

riscv:
	cd $(test_dir)/riscv-mini && lit -v . --path=$(root_dir)/utils/bin/

units = ALUTop Datapath Control Core Test
v     = $(addsuffix .fir.v, $(units))

$(units): % :
	firrtl -X verilog -i test/chisel3/$*.fir -o test/chisel3/$*.fir.v -p c > test/chisel3/$*.fir.out 
	#scp test/chisel3/$*.fir.v adamiz@a5:/scratch/adamiz/firrtl-all/riscv-mini/generated-src/$*.v

done: build-fast check regress
	say "done"

fail:
	say "fail"

# Scala Added Makefile commands

build-scala:
	sbt "assembly"

test-scala:
	cd $(test_dir)/parser && lit -v . --path=$(root_dir)/utils/bin/
	cd $(test_dir)/passes/infer-types && lit -v . --path=$(root_dir)/utils/bin/

set-scala:
	ln -f -s $(root_dir)/utils/bin/firrtl-scala $(root_dir)/utils/bin/firrtl

set-stanza:
	ln -f -s $(root_dir)/utils/bin/firrtl-stanza $(root_dir)/utils/bin/firrtl

.PHONY: all install build-deploy build check clean fail succeed regress set-scala set-stanza build-scala test-scala
