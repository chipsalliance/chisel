root_dir ?= $(PWD)

SBT ?= sbt
SBT_FLAGS ?= -Dsbt.log.noformat=true
MKDIR ?= mkdir -p
CURL ?= curl -L
MILL_BIN ?= $(HOME)/bin/mill
MILL ?= $(MILL_BIN) --color false

# Ensure the default target is something reasonable.
default: build

# Fetch mill (if we don't have it.
$(MILL_BIN):
	$(MKDIR) $(dir $@)
	$(CURL) -o $@ https://github.com/ucbjrl/mill/releases/download/v0.2.0-FDF/mill-0.2.0-FDF && chmod +x $@

mill-tools:	$(MILL_BIN)

# Compile and package jar
mill.build: mill-tools
	$(MILL) firrtlInterpreter.jar

# Compile and test
mill.test: mill-tools
	$(MILL) firrtlInterpreter.test

# Build and publish jar
mill.publishLocal: mill-tools
	$(MILL) firrtlInterpreter.publishLocal

# Compile and package all jar
mill.build.all: mill-tools
	$(MILL) firrtlInterpreter[_].jar

# Compile and test
mill.test.all: mill-tools
	$(MILL) firrtlInterpreter[_].test

# Build and publish jar
mill.publishLocal.all: mill-tools
	$(MILL) firrtlInterpreter[_].publishLocal

# Remove all generated code.
# Until "mill clean" makes it into a release.
mill.clean:
	$(RM) -rf out

clean:	mill.clean
	$(SBT) "+clean"

build:	mill.build

.PHONY: build clean mill.build mill.test mill.publishLocal mill.build.all mill.test.all mill.publishLocal.all
