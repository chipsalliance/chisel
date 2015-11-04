# Retain all intermediate files.
.SECONDARY:

SBT		?= sbt
SBT_FLAGS	?= -Dsbt.log.noformat=true

CHISEL_VERSION = $(shell "$(SBT)" $(SBT_FLAGS) "show version" | tail -n 1 | cut -d ' ' -f 2)

SRC_DIR	?= .
CHISEL_BIN ?= $(abspath $(SRC_DIR)/bin)
export CHISEL_BIN

MV	?= mv
FIRRTL	?= firrtl

#$(info Build Chisel $(CHISEL_VERSION))

# The targetDir will be rm -rf'ed when "make clean"
targetDir ?= ./generated
# The TEST_OUTPUT_DIR will be rm -rf'ed when "make clean"
TEST_OUTPUT_DIR ?= ./test-outputs
RM_DIRS 	:= $(TEST_OUTPUT_DIR) test-reports $(targetDir)
#CLEAN_DIRS	:= doc

test_src_dir := src/test/scala/chiselTests
test_results := $(filter BundleWire ComplexAssign GCD MulLookup Stack Tbl,$(notdir $(basename $(wildcard $(test_src_dir)/*.scala))))
c_resources_dir := src/main/resources

test_vs    := $(addprefix $(targetDir)/, $(addsuffix .v, $(test_results)))

.PHONY:	smoke publish-local check clean jenkins-build coverage scaladoc test checkstyle compile

default:	publish-local

smoke compile:
	$(SBT) $(SBT_FLAGS) compile

publish-local:
	$(SBT) $(SBT_FLAGS) publish-local

test:
	$(SBT) $(SBT_FLAGS) test

check:	test $(test_vs)

checkstyle:
	$(SBT) $(SBT_FLAGS) scalastyle test:scalastyle

coverage:
	$(SBT) $(SBT_FLAGS) coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

clean:
	$(SBT) $(SBT_FLAGS) clean
ifneq (,$(CLEAN_DIRS))
	for dir in $(CLEAN_DIRS); do $(MAKE) -C $$dir clean; done
endif
ifneq (,$(RM_DIRS))
	$(RM) -r $(RM_DIRS)
endif

scaladoc:
	$(SBT) $(SBT_FLAGS) doc

site:
	$(SBT) $(SBT_FLAGS) make-site

# Start off clean, then run tests for all supported configurations, and publish those versions of the code.
# Then run coverage and style tests (for developer's use).
# Don't publish the coverage test code since it contains hooks/references to the coverage test package
# and we don't want code with those dependencies published.
# We need to run the coverage tests last, since Jenkins will fail the build if it can't find their results.
jenkins-build: clean
	$(SBT) $(SBT_FLAGS) test
	$(SBT) $(SBT_FLAGS) clean publish-local
	$(SBT) $(SBT_FLAGS) scalastyle coverage test
	$(SBT) $(SBT_FLAGS) coverageReport

$(targetDir)/%.fir: $(test_src_dir)/%.scala
	$(SBT) $(SBT_FLAGS) "test:runMain chiselTests.MiniChisel $(notdir $(basename $<)) --targetDir $(targetDir) $(CHISEL_FLAGS)"

$(targetDir)/%.v: $(targetDir)/%.fir
	$(FIRRTL) -i $< -o $@ -X verilog
