# Installs stanza into /usr/local/bin
# TODO Talk to Patrick to fill this in

root_dir ?= $(PWD)
test_dir ?= $(root_dir)/test
firrtl_dir ?= $(root_dir)/src/main/stanza

all-noise: 
	${MAKE} all || ${MAKE} fail

all: done

install-linux:
	cd src/lib && unzip stanza-linux.zip
	cd src/lib/stanza && sudo ./stanza -platform linux -install /usr/local/bin/stanza

install-mac:
	cd src/lib && unzip stanza-mac.zip
	cd src/lib/stanza && sudo ./stanza -platform os-x -install /usr/local/bin/stanza

build-deploy: 
	cd $(firrtl_dir) && stanza -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl

build: 
	cd $(firrtl_dir) && stanza -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl

build-fast: 
	cd $(firrtl_dir) && stanza -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl -flags OPTIMIZE

check:
	cd $(test_dir) && lit -v . --path=$(root_dir)/utils/bin/

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

riscv:
	cd $(test_dir)/riscv-mini && lit -v . --path=$(root_dir)/utils/bin/

units = ALUTop Datapath Control Core Test
v     = $(addsuffix .fir.v, $(units))

$(units): % :
	firrtl -X verilog -i test/chisel3/$*.fir -o test/chisel3/$*.fir.v -p c > test/chisel3/$*.fir.out 
	#scp test/chisel3/$*.fir.v adamiz@a5:/scratch/adamiz/firrtl-all/riscv-mini/generated-src/$*.v

done: build check
	say "done"

fail:
	say "fail"

.PHONY: all install build-deploy build check clean fail succeed
