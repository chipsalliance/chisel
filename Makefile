root_dir ?= $(PWD)
regress_dir ?= $(root_dir)/regress
install_dir ?= $(root_dir)/utils/bin

SBT ?= sbt
SBT_FLAGS ?= -Dsbt.log.noformat=true

scala_jar ?= $(install_dir)/firrtl.jar
scala_src := $(shell find src -type f \( -name "*.scala" -o -path "*/resources/*" \))

clean:
	$(MAKE) -C $(root_dir)/spec clean
	rm -f $(install_dir)/firrtl.jar
	$(SBT) "clean"

.PHONY : specification
specification:
	$(MAKE) -C $(root_dir)/spec all

build:	build-scala

regress: $(scala_jar)
	cd $(regress_dir) && $(install_dir)/firrtl -i rocket.fir -o rocket.v -X verilog

# Scala Added Makefile commands

build-scala: $(scala_jar)

$(scala_jar): $(scala_src)
	$(SBT) "assembly"

test-scala:
	$(SBT) test

jenkins-build:	clean
	$(SBT) $(SBT_FLAGS) +clean +test +publish-local
	$(SBT) $(SBT_FLAGS) scalastyle coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

.PHONY: build clean regress build-scala test-scala
