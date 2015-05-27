# Installs stanza into /usr/local/bin
# TODO Talk to Patrick to fill this in

root_dir ?= $(PWD)
test_dir ?= $(root_dir)/test
firrtl_dir ?= $(root_dir)/src/main/stanza

all-noise: 
	${MAKE} all || ${MAKE} fail

all: build check done

install:
	cd src/lib && unzip stanza.zip
	cd src/lib/stanza && sudo ./stanza -platform os-x -install /usr/local/bin/stanza

build-deploy: 
	cd $(firrtl_dir) && stanza -i firrtl-main.stanza -o $(root_dir)/utils/bin/firrtl

build: 
	cd $(firrtl_dir) && stanza -i firrtl-test-main.stanza -o $(root_dir)/utils/bin/firrtl

check: 
	cd $(test_dir) && lit -v . --path=$(root_dir)/utils/bin/

passes: 
	cd $(test_dir)/passes && lit -v . --path=$(root_dir)/utils/bin/

errors:
	cd $(test_dir)/errors && lit -v . --path=$(root_dir)/utils/bin/

chisel3:
	cd $(test_dir)/chisel3 && lit -v . --path=$(root_dir)/utils/bin/

features:
	cd $(test_dir)/features && lit -v . --path=$(root_dir)/utils/bin/

clean:
	rm -f $(test_dir)/*/*/*.out
	rm -f $(test_dir)/*/*.out

done:
	say "done"

fail:
	say "fail"

.PHONY: all install build-deploy build check clean fail succeed
