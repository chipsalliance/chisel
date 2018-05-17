root_dir ?= $(PWD)
regress_dir ?= $(root_dir)/regress
install_dir ?= $(root_dir)/utils/bin

SBT ?= sbt
SBT_FLAGS ?= -Dsbt.log.noformat=true
MKDIR ?= mkdir -p
CURL ?= curl -L
MILL_BIN ?= $(HOME)/bin/mill
# The following must correspond with the equivalent definition in build.sc
ANTLR4_JAR ?= $(HOME)/lib/antlr-4.7.1-complete.jar
MILL ?= $(MILL_BIN) --color false

# Ensure the default target is something reasonable.
default: build

# Fetch mill (if we don't have it.
$(MILL_BIN):
	$(MKDIR) $(dir $@)
	$(CURL) -o $@ https://github.com/ucbjrl/mill/releases/download/v0.2.0-FDF/mill-0.2.0-FDF && chmod +x $@

# Fetch antlr4 (if we don't have it.
$(ANTLR4_JAR):
	$(MKDIR) $(dir $@)
	$(CURL) -o $@ https://www.antlr.org/download/antlr-4.7.1-complete.jar	
mill-tools:	$(MILL_BIN) $(ANTLR4_JAR)

# Compile and package jar
mill.build: mill-tools
	$(MILL) firrtl.jar

# Compile and test
mill.test: mill-tools
	$(MILL) firrtl.test

# Build and publish jar
mill.publishLocal: mill-tools
	$(MILL) firrtl.publishLocal

# Compile and package all jar
mill.build.all: mill-tools
	$(MILL) firrtl[_].jar

# Compile and test
mill.test.all: mill-tools
	$(MILL) firrtl[_].test

# Build and publish jar
mill.publishLocal.all: mill-tools
	$(MILL) firrtl[_].publishLocal

# Remove all generated code.
# Until "mill clean" makes it into a release.
mill.clean:
	$(RM) -rf out

scala_jar ?= $(install_dir)/firrtl.jar
scala_src := $(shell find src -type f \( -name "*.scala" -o -path "*/resources/*" \))

clean:
	rm -f $(install_dir)/firrtl.jar
	$(SBT) "clean"

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
