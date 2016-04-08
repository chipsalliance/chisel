root_dir ?= $(PWD)
test_dir ?= $(root_dir)/test
regress_dir ?= $(root_dir)/regress
firrtl_dir ?= $(root_dir)/src/main/stanza
install_dir ?= $(root_dir)/utils/bin

sbt ?= sbt
stanza ?= $(install_dir)/stanza
stanza_bin ?= $(install_dir)/firrtl-stanza
scala_jar ?= $(install_dir)/firrtl.jar
scala_src := $(shell find src -type f \( -name "*.scala" -o -path "*/resources/*" \))
stanza_src=$(shell ls src/main/stanza/*.stanza)

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

utils/bin/stanza: $(stanza)
$(stanza): $(root_dir)/src/lib/stanza/stamp $(root_dir)/utils/stanza-wrapper
	cd src/lib/stanza && ./stanza -platform $(stanza_target_name) -install $(stanza)
	cat $(root_dir)/utils/stanza-wrapper | sed 's!@@TOP@@!$(root_dir)!g' > $@

$(stanza_bin): $(stanza) $(stanza_src)
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $@

build-stanza: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(install_dir)/firrtl-stanza
	$(MAKE) set-stanza

build-fast: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-test-main.stanza -o $(install_dir)/firrtl-stanza -flags OPTIMIZE
	$(MAKE) set-stanza

build-deploy: $(stanza)
	cd $(firrtl_dir) && $(stanza) -i firrtl-main.stanza -o $(install_dir)/firrtl-stanza
	$(MAKE) set-stanza

check: 
	cd $(test_dir) && lit -j 2 -v . --path=$(install_dir)/

regress: 
	cd $(regress_dir) && $(install_dir)/firrtl -i rocket.fir -o rocket.v -X verilog

parser: 
	cd $(test_dir)/parser && lit -v . --path=$(install_dir)/

perf: 
	cd $(test_dir)/performance && lit -v . --path=$(install_dir)/

jack: 
	cd $(test_dir)/passes/jacktest && lit -v . --path=$(install_dir)/

passes: 
	cd $(test_dir)/passes && lit -v . --path=$(install_dir)/

errors:
	cd $(test_dir)/errors && lit -v . --path=$(install_dir)/

features:
	cd $(test_dir)/features && lit -j 2 -v . --path=$(install_dir)/

chirrtl:
	cd $(test_dir)/chirrtl && lit -v . --path=$(install_dir)/

custom:
	cd $(test_dir)/custom && lit -v . --path=$(install_dir)/ --max-time=10

clean:
	rm -f $(test_dir)/*/*/*.out
	rm -f $(test_dir)/*/*.out
	rm -rf src/lib/stanza
	rm -f $(stanza)
	rm -f $(install_dir)/firrtl.jar
	rm -f $(install_dir)/firrtl
	rm -f $(install_dir)/firrtl-stanza
	"$(sbt)" "clean"

riscv:
	cd $(test_dir)/riscv-mini && lit -v . --path=$(install_dir)/

units = ALUTop Datapath Control Core Test
v     = $(addsuffix .fir.v, $(units))

$(units): % :
	firrtl -X verilog -i test/chisel3/$*.fir -o test/chisel3/$*.fir.v -p c > test/chisel3/$*.fir.out 
	#scp test/chisel3/$*.fir.v adamiz@a5:/scratch/adamiz/firrtl-all/riscv-mini/generated-src/$*.v

done: build-fast check regress
	say "done"

fail:
	say "fail"

build:	build-scala

# Scala Added Makefile commands

build-scala: $(scala_jar)
	$(MAKE) set-scala

$(scala_jar): $(scala_src)
	"$(sbt)" "assembly"

test-scala:
	"$(sbt)" test

set-scala:
	ln -f -s $(install_dir)/firrtl-scala $(install_dir)/firrtl

set-stanza:
	ln -f -s $(install_dir)/firrtl-stanza $(install_dir)/firrtl

set-linux:
	ln -f -s $(install_dir)/FileCheck_linux $(install_dir)/FileCheck

set-osx:
	ln -f -s $(install_dir)/FileCheck_mac $(install_dir)/FileCheck

.PHONY: all install build-deploy build check clean fail succeed regress set-scala set-stanza build-scala test-scala
