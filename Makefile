SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true
MKDIR ?= mkdir -p
CURL ?= curl -L
MILL_BIN ?= $(HOME)/bin/mill
MILL ?= $(MILL_BIN) --color false
MILL_REMOTE_RELEASE ?= https://github.com/lihaoyi/mill/releases/download/0.3.5/0.3.5

# Fetch mill (if we don't have it).
$(MILL_BIN):
	$(MKDIR) $(dir $@)
	@echo $(CURL) --silent --output $@.curl --write-out "%{http_code}" $(MILL_REMOTE_RELEASE)
	STATUSCODE=$(shell $(CURL) --silent --output $@.curl --write-out "%{http_code}" $(MILL_REMOTE_RELEASE)) && \
	if test $$STATUSCODE -eq 200; then \
	  mv $@.curl $@ && chmod +x $@ ;\
	else \
	  echo "Can't fetch $(MILL_REMOTE_RELEASE)" && cat $@.curl && echo ;\
	  false ;\
	fi

mill-tools:	$(MILL_BIN)

CHISEL_VERSION = $(shell "$(SBT)" $(SBT_FLAGS) "show version" | tail -n 1 | cut -d ' ' -f 2)

#$(info Build Chisel $(CHISEL_VERSION))

# The TEST_OUTPUT_DIR will be rm -rf'ed when "make clean"
TEST_OUTPUT_DIR ?= ./test_run_dur
RM_DIRS 	:= $(TEST_OUTPUT_DIR)

.PHONY:	smoke publish-local pubishLocal check clean jenkins-build coverage scaladoc test compile \
	mill.build mill.test mill.publishLocal mill.build.all mill.test.all mill.publishLocal.all mill-tools

default:	publishLocal

smoke compile:
	$(SBT) $(SBT_FLAGS) compile

publish-local publishLocal:
	$(SBT) $(SBT_FLAGS) publishLocal

test:
	$(SBT) $(SBT_FLAGS) test

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport coverageAggregate

clean:
	$(SBT) $(SBT_FLAGS) clean
ifneq (,$(CLEAN_DIRS))
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
endif
ifneq (,$(RM_DIRS))
	$(RM) -r $(RM_DIRS)
endif

scaladoc:
	$(SBT) $(SBT_FLAGS) unidoc

site:
	$(SBT) $(SBT_FLAGS) make-site

# Start off clean, then run tests for all supported configurations, and publish those versions of the code.
# Then run coverage and style tests (for developer's use).
# Don't publish the coverage test code since it contains hooks/references to the coverage test package
# and we don't want code with those dependencies published.
# We need to run the coverage tests last, since Jenkins will fail the build if it can't find their results.
jenkins-build: clean
	$(SBT) $(SBT_FLAGS) test
	$(SBT) $(SBT_FLAGS) clean publishLocal
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

# Compile and package jar
mill.build: mill-tools
	$(MILL) chisel3.jar

# Compile and test
mill.test: mill-tools
	$(MILL) chisel3.test

# Build and publish jar
mill.publishLocal: mill-tools
	$(MILL) chisel3.publishLocal

# Compile and package all jar
mill.build.all: mill-tools
	$(MILL) chisel3[_].jar

# Compile and test
mill.test.all: mill-tools
	$(MILL) chisel3[_].test

# Build and publish jar
mill.publishLocal.all: mill-tools
	$(MILL) chisel3[_].publishLocal

# Remove all generated code.
# Until "mill clean" makes it into a release.
mill.clean:
	$(RM) -rf out

# The "last-resort" rule.
# We assume the target is something like "+clean".
%::
	$(SBT) $(SBT_FLAGS) $@
